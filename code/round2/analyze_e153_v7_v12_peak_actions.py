#!/usr/bin/env python3
"""Audit the PAS/PDP peak actions that turn the V7 channel into V12."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from scipy.ndimage import maximum_filter, maximum_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

import analyze_e138_v12_guided_official_correction_student as e138
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META, statistics
from run_v6_regime_component_lifecycle import REGIME_CACHE, REGIME_NAMES
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E153-v7-v12-peak-action-audit"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v7_v12_peak_actions"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v7_v12_peak_actions"
VISUAL_DIR = PATHS.analysis / "visualizations" / "复赛V7与V12峰动作分析"
REPORT = ANALYSIS_DIR / "v7_v12_peak_actions.json"
REPORT_MD = ANALYSIS_DIR / "V7与V12峰动作数据分析.md"
POINT_CSV = ANALYSIS_DIR / "500点修正动作明细.csv"
EVENT_CSV = ANALYSIS_DIR / "角度与时延峰动作明细.csv"
CORRELATION_CSV = ANALYSIS_DIR / "信息与修正动作相关性.csv"
PREDICTABILITY_CSV = ANALYSIS_DIR / "信息组跨空间区域解释力.csv"
METRICS = ARTIFACT_DIR / "v7_v12_peak_action_metrics.npz"

V7_CHANNEL = e138.V7_CHANNEL
V12_CHANNEL = e138.V12_CHANNEL
TRAIN_CHANNEL = e138.TRAIN_CHANNEL
GEOMETRY = e138.GEOMETRY
ANCHOR_PAS = e138.OFFICIAL_ANCHOR_PAS
ANCHOR_PDP = e138.OFFICIAL_ANCHOR_PDP
CARRIER_BANK = (
    PATHS.artifacts / "round2" / "complex_carrier_bank" / "oof_complex_carrier_bank.npy"
)
CARRIER_META = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_bank"
    / "complex_carrier_bank_metadata.json"
)
NEIGHBOR_CACHE = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_oof_weights.npz"
)

ANGLE_PEAK_MIN = 0.010
ANGLE_MAX_PEAKS = 8
ANGLE_MATCH_H = 2.0
ANGLE_MATCH_V = 1.0
ANGLE_MASS_ABS_TOL = 0.002
DELAY_PEAK_MIN = 0.001
DELAY_MAX_PEAKS = 8
DELAY_MATCH_BINS = 3.0
DELAY_MASS_ABS_TOL = 0.001
TINY = 1e-30

MAP_LABELS = {
    "center_hit_count": "中心射线遮挡次数",
    "center_first_fraction": "首次遮挡相对位置",
    "center_last_fraction": "末次遮挡相对位置",
    "center_blocked_span": "中心射线遮挡跨度",
    "fan_hit_mean": "扇形射线平均遮挡数",
    "fan_hit_min": "扇形射线最少遮挡数",
    "fan_hit_max": "扇形射线最多遮挡数",
    "fan_clear_fraction": "扇形射线可见比例",
    "fan_first_mean": "扇形首次遮挡均值",
    "fan_first_min": "扇形最早遮挡位置",
    "fan_last_mean": "扇形末次遮挡均值",
    "fan_span_mean": "扇形遮挡跨度均值",
    "fan_hit_std": "扇形遮挡数波动",
    "surface_nearest_xy": "最近表面距离",
    "surface_mean16_xy": "邻近表面平均距离",
    "surface_mean_height": "邻近表面平均高度",
    "surface_max_height": "邻近表面最大高度",
    "surface_height_std": "邻近表面高度波动",
    "surface_vertical_fraction": "竖直表面比例",
    "wall_nearest_xy": "最近墙面距离",
    "wall_mean8_xy": "邻近墙面平均距离",
    "wall_mean_height": "邻近墙面平均高度",
    "wall_max_height": "邻近墙面最大高度",
}

TARGET_LABELS = {
    "angle_action": "角度修正幅度",
    "delay_action": "时延修正幅度",
    "pas_gain": "PAS得分增益",
    "pdp_gain": "PDP得分增益",
    "angle_new_mass": "角度新增峰质量",
    "delay_new_mass": "时延新增峰质量",
}


@dataclass(frozen=True)
class Peak:
    coordinate: tuple[int, ...]
    value: float
    mass: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trees", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20_260_810_153)
    return parser.parse_args()


def configure_chinese_font() -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    candidates = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Droid Sans Fallback",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    )
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    raise RuntimeError("未找到可用的中文字体")


def normalize(value: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axis, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, TINY),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def channel_group_maps(
    channel: np.ndarray,
    device: torch.device,
    batch_size: int,
    query_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(query_ids) if query_ids is not None else len(channel)
    angle = np.empty((count, e138.N_UE, e138.MH, e138.MV), dtype=np.float32)
    delay = np.empty((count, e138.MP, e138.N_UE, e138.S_SC), dtype=np.float32)
    log_power = np.empty(count, dtype=np.float32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        if query_ids is None:
            source = channel[start:stop]
        else:
            source = channel[query_ids[start:stop]]
        value = torch.as_tensor(
            np.array(source, dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            shaped = value.reshape(
                len(value), e138.MP, e138.MH, e138.MV, e138.N_UE, e138.S_SC
            )
            pas = torch.fft.fft2(shaped, dim=(2, 3)).abs().square()
            pas = pas / pas.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
            angle[start:stop] = (
                pas.sum(dim=1).mean(dim=-1).permute(0, 3, 1, 2).cpu().numpy()
            )

            pdp = torch.fft.ifft(value, dim=-1).abs().square().reshape(
                len(value), e138.MP, e138.MH, e138.MV, e138.N_UE, e138.S_SC
            )
            pdp = pdp / pdp.sum(dim=-1, keepdim=True).clamp_min(TINY)
            delay[start:stop] = pdp.mean(dim=(2, 3)).cpu().numpy()
            power = value.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
            log_power[start:stop] = torch.log10(power.clamp_min(TINY)).cpu().numpy()
    return normalize(angle, (2, 3)), normalize(delay, 3), log_power


def channel_point_metrics(
    prediction: np.ndarray,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = [np.empty(len(query_ids), dtype=np.float64) for _ in range(4)]
    for start in range(0, len(query_ids), batch_size):
        stop = min(start + batch_size, len(query_ids))
        pred = torch.as_tensor(
            np.array(prediction[start:stop], dtype=np.complex64, copy=True),
            device=device,
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:stop]], dtype=np.complex64, copy=True),
            device=device,
        )
        with torch.inference_mode():
            point = point_metrics(
                pred, truth, pas_distribution(truth), pdp_distribution(truth)
            )
        for destination, source in zip(values, point, strict=True):
            destination[start:stop] = source
    return tuple(values)


def entropy(probability: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod([probability.shape[axis] for axis in axes]))
    value = -np.sum(
        probability * np.log(np.maximum(probability, 1e-12)),
        axis=axes,
        dtype=np.float64,
    )
    return value / math.log(size)


def top_mass(probability: np.ndarray, axes: tuple[int, ...], count: int) -> np.ndarray:
    untouched = tuple(axis for axis in range(probability.ndim) if axis not in axes)
    moved = np.transpose(probability, (*untouched, *axes))
    flat = moved.reshape(*moved.shape[: len(untouched)], -1)
    take = min(count, flat.shape[-1])
    indices = np.argpartition(flat, -take, axis=-1)[..., -take:]
    return np.take_along_axis(flat, indices, axis=-1).sum(axis=-1, dtype=np.float64)


def hellinger_squared(left: np.ndarray, right: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    coefficient = np.sum(
        np.sqrt(np.maximum(left * right, 0.0)), axis=axis, dtype=np.float64
    )
    return np.maximum(0.0, 1.0 - coefficient)


def direction_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    b = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    numerator = np.sum(a * b, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(a * a, axis=1, dtype=np.float64)
        * np.sum(b * b, axis=1, dtype=np.float64)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > TINY,
    )


def angle_window_mass(value: np.ndarray, h: int, v: int) -> float:
    indices_h = [(h + offset) % e138.MH for offset in (-1, 0, 1)]
    indices_v = [(v + offset) % e138.MV for offset in (-1, 0, 1)]
    return float(value[np.ix_(indices_h, indices_v)].sum(dtype=np.float64))


def delay_window_mass(value: np.ndarray, delay: int) -> float:
    indices = [(delay + offset) % e138.S_SC for offset in range(-2, 3)]
    return float(value[indices].sum(dtype=np.float64))


def angle_window_mask(h: int, v: int) -> np.ndarray:
    output = np.zeros((e138.MH, e138.MV), dtype=bool)
    for dh in (-1, 0, 1):
        for dv in (-1, 0, 1):
            output[(h + dh) % e138.MH, (v + dv) % e138.MV] = True
    return output


def delay_window_mask(delay: int) -> np.ndarray:
    output = np.zeros(e138.S_SC, dtype=bool)
    for offset in range(-2, 3):
        output[(delay + offset) % e138.S_SC] = True
    return output


def extract_angle_peaks(value: np.ndarray) -> list[Peak]:
    if float(value.sum()) <= TINY:
        return []
    local = maximum_filter(value, size=(3, 3), mode="wrap")
    candidates = np.argwhere((value >= local - 1e-12) & (value >= ANGLE_PEAK_MIN))
    if len(candidates) == 0:
        candidates = np.asarray([np.unravel_index(np.argmax(value), value.shape)])
    order = np.argsort([-float(value[tuple(index)]) for index in candidates])
    peaks = []
    for local_index in order[:ANGLE_MAX_PEAKS]:
        h, v = (int(item) for item in candidates[local_index])
        peaks.append(
            Peak((h, v), float(value[h, v]), angle_window_mass(value, h, v))
        )
    return peaks


def extract_delay_peaks(value: np.ndarray) -> list[Peak]:
    if float(value.sum()) <= TINY:
        return []
    local = maximum_filter1d(value, size=5, mode="wrap")
    candidates = np.flatnonzero((value >= local - 1e-12) & (value >= DELAY_PEAK_MIN))
    if len(candidates) == 0:
        candidates = np.asarray([int(np.argmax(value))])
    order = candidates[np.argsort(-value[candidates])][:DELAY_MAX_PEAKS]
    return [
        Peak((int(index),), float(value[index]), delay_window_mass(value, int(index)))
        for index in order
    ]


def periodic_axis_distance(left: int, right: int, period: int) -> float:
    delta = abs(left - right)
    return float(min(delta, period - delta))


def angle_distance(left: Peak, right: Peak) -> tuple[float, float, float]:
    dh = periodic_axis_distance(left.coordinate[0], right.coordinate[0], e138.MH)
    dv = periodic_axis_distance(left.coordinate[1], right.coordinate[1], e138.MV)
    cost = math.sqrt((dh / ANGLE_MATCH_H) ** 2 + (dv / ANGLE_MATCH_V) ** 2)
    return cost, dh, dv


def delay_distance(left: Peak, right: Peak) -> float:
    return periodic_axis_distance(
        left.coordinate[0], right.coordinate[0], e138.S_SC
    )


def mass_action(old: float, new: float, absolute_tolerance: float) -> str:
    tolerance = max(absolute_tolerance, 0.05 * max(old, new))
    if new - old > tolerance:
        return "增强"
    if old - new > tolerance:
        return "减弱"
    return "基本不变"


def local_error_reduction(
    base: np.ndarray,
    changed: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> float:
    before = np.abs(base[mask] - truth[mask]).sum(dtype=np.float64)
    after = np.abs(changed[mask] - truth[mask]).sum(dtype=np.float64)
    return float(before - after)


def nearest_truth_angle_distance(peak: Peak, truth_peaks: list[Peak]) -> float:
    if not truth_peaks:
        return float("nan")
    return float(min(angle_distance(peak, target)[0] for target in truth_peaks))


def nearest_truth_delay_distance(peak: Peak, truth_peaks: list[Peak]) -> float:
    if not truth_peaks:
        return float("nan")
    return float(min(delay_distance(peak, target) for target in truth_peaks))


def match_angle_events(
    base: np.ndarray,
    changed: np.ndarray,
    truth: np.ndarray,
    point_row: int,
    query_id: int,
    site: int,
    ue: int,
) -> list[dict[str, object]]:
    old_peaks = extract_angle_peaks(base)
    new_peaks = extract_angle_peaks(changed)
    truth_peaks = extract_angle_peaks(truth)
    accepted: list[tuple[int, int, float, float, float]] = []
    if old_peaks and new_peaks:
        cost = np.empty((len(old_peaks), len(new_peaks)), dtype=np.float64)
        axis = {}
        for old_index, old in enumerate(old_peaks):
            for new_index, new in enumerate(new_peaks):
                value, dh, dv = angle_distance(old, new)
                cost[old_index, new_index] = value
                axis[(old_index, new_index)] = (dh, dv)
        rows, columns = linear_sum_assignment(cost)
        for old_index, new_index in zip(rows, columns, strict=True):
            if cost[old_index, new_index] <= 1.0:
                dh, dv = axis[(int(old_index), int(new_index))]
                accepted.append(
                    (int(old_index), int(new_index), float(cost[old_index, new_index]), dh, dv)
                )
    old_matched = {item[0] for item in accepted}
    new_matched = {item[1] for item in accepted}
    events: list[dict[str, object]] = []

    def common() -> dict[str, object]:
        return {
            "domain": "角度",
            "point_row": point_row,
            "query_id": query_id,
            "site": site,
            "group": f"UE{ue}",
            "polarization": -1,
            "ue": ue,
        }

    for old_index, new_index, cost, dh, dv in accepted:
        old = old_peaks[old_index]
        new = new_peaks[new_index]
        mask = angle_window_mask(*old.coordinate) | angle_window_mask(*new.coordinate)
        reduction = local_error_reduction(base, changed, truth, mask)
        position = "移动" if dh > 0.0 or dv > 0.0 else "保留"
        events.append(
            {
                **common(),
                "position_action": position,
                "mass_action": mass_action(old.mass, new.mass, ANGLE_MASS_ABS_TOL),
                "old_h": old.coordinate[0],
                "old_v": old.coordinate[1],
                "old_delay": -1,
                "new_h": new.coordinate[0],
                "new_v": new.coordinate[1],
                "new_delay": -1,
                "shift_h": dh,
                "shift_v": dv,
                "shift_delay": 0.0,
                "match_cost": cost,
                "old_peak_value": old.value,
                "new_peak_value": new.value,
                "old_mass": old.mass,
                "new_mass": new.mass,
                "mass_delta": new.mass - old.mass,
                "truth_distance_old": nearest_truth_angle_distance(old, truth_peaks),
                "truth_distance_new": nearest_truth_angle_distance(new, truth_peaks),
                "local_error_reduction": reduction,
                "beneficial": reduction > 0.0,
            }
        )
    for old_index, old in enumerate(old_peaks):
        if old_index in old_matched:
            continue
        mask = angle_window_mask(*old.coordinate)
        reduction = local_error_reduction(base, changed, truth, mask)
        events.append(
            {
                **common(),
                "position_action": "消失",
                "mass_action": "减弱",
                "old_h": old.coordinate[0],
                "old_v": old.coordinate[1],
                "old_delay": -1,
                "new_h": -1,
                "new_v": -1,
                "new_delay": -1,
                "shift_h": float("nan"),
                "shift_v": float("nan"),
                "shift_delay": float("nan"),
                "match_cost": float("nan"),
                "old_peak_value": old.value,
                "new_peak_value": 0.0,
                "old_mass": old.mass,
                "new_mass": 0.0,
                "mass_delta": -old.mass,
                "truth_distance_old": nearest_truth_angle_distance(old, truth_peaks),
                "truth_distance_new": float("nan"),
                "local_error_reduction": reduction,
                "beneficial": reduction > 0.0,
            }
        )
    for new_index, new in enumerate(new_peaks):
        if new_index in new_matched:
            continue
        mask = angle_window_mask(*new.coordinate)
        reduction = local_error_reduction(base, changed, truth, mask)
        events.append(
            {
                **common(),
                "position_action": "新增",
                "mass_action": "增强",
                "old_h": -1,
                "old_v": -1,
                "old_delay": -1,
                "new_h": new.coordinate[0],
                "new_v": new.coordinate[1],
                "new_delay": -1,
                "shift_h": float("nan"),
                "shift_v": float("nan"),
                "shift_delay": float("nan"),
                "match_cost": float("nan"),
                "old_peak_value": 0.0,
                "new_peak_value": new.value,
                "old_mass": 0.0,
                "new_mass": new.mass,
                "mass_delta": new.mass,
                "truth_distance_old": float("nan"),
                "truth_distance_new": nearest_truth_angle_distance(new, truth_peaks),
                "local_error_reduction": reduction,
                "beneficial": reduction > 0.0,
            }
        )
    return events


def match_delay_events(
    base: np.ndarray,
    changed: np.ndarray,
    truth: np.ndarray,
    point_row: int,
    query_id: int,
    site: int,
    polarization: int,
    ue: int,
) -> list[dict[str, object]]:
    old_peaks = extract_delay_peaks(base)
    new_peaks = extract_delay_peaks(changed)
    truth_peaks = extract_delay_peaks(truth)
    accepted: list[tuple[int, int, float]] = []
    if old_peaks and new_peaks:
        cost = np.empty((len(old_peaks), len(new_peaks)), dtype=np.float64)
        for old_index, old in enumerate(old_peaks):
            for new_index, new in enumerate(new_peaks):
                cost[old_index, new_index] = delay_distance(old, new)
        rows, columns = linear_sum_assignment(cost)
        for old_index, new_index in zip(rows, columns, strict=True):
            if cost[old_index, new_index] <= DELAY_MATCH_BINS:
                accepted.append(
                    (int(old_index), int(new_index), float(cost[old_index, new_index]))
                )
    old_matched = {item[0] for item in accepted}
    new_matched = {item[1] for item in accepted}
    events: list[dict[str, object]] = []

    def common() -> dict[str, object]:
        return {
            "domain": "时延",
            "point_row": point_row,
            "query_id": query_id,
            "site": site,
            "group": f"P{polarization}-UE{ue}",
            "polarization": polarization,
            "ue": ue,
        }

    for old_index, new_index, shift in accepted:
        old = old_peaks[old_index]
        new = new_peaks[new_index]
        mask = delay_window_mask(old.coordinate[0]) | delay_window_mask(
            new.coordinate[0]
        )
        reduction = local_error_reduction(base, changed, truth, mask)
        events.append(
            {
                **common(),
                "position_action": "移动" if shift > 0.0 else "保留",
                "mass_action": mass_action(old.mass, new.mass, DELAY_MASS_ABS_TOL),
                "old_h": -1,
                "old_v": -1,
                "old_delay": old.coordinate[0],
                "new_h": -1,
                "new_v": -1,
                "new_delay": new.coordinate[0],
                "shift_h": 0.0,
                "shift_v": 0.0,
                "shift_delay": shift,
                "match_cost": shift,
                "old_peak_value": old.value,
                "new_peak_value": new.value,
                "old_mass": old.mass,
                "new_mass": new.mass,
                "mass_delta": new.mass - old.mass,
                "truth_distance_old": nearest_truth_delay_distance(old, truth_peaks),
                "truth_distance_new": nearest_truth_delay_distance(new, truth_peaks),
                "local_error_reduction": reduction,
                "beneficial": reduction > 0.0,
            }
        )
    for old_index, old in enumerate(old_peaks):
        if old_index in old_matched:
            continue
        mask = delay_window_mask(old.coordinate[0])
        reduction = local_error_reduction(base, changed, truth, mask)
        events.append(
            {
                **common(),
                "position_action": "消失",
                "mass_action": "减弱",
                "old_h": -1,
                "old_v": -1,
                "old_delay": old.coordinate[0],
                "new_h": -1,
                "new_v": -1,
                "new_delay": -1,
                "shift_h": float("nan"),
                "shift_v": float("nan"),
                "shift_delay": float("nan"),
                "match_cost": float("nan"),
                "old_peak_value": old.value,
                "new_peak_value": 0.0,
                "old_mass": old.mass,
                "new_mass": 0.0,
                "mass_delta": -old.mass,
                "truth_distance_old": nearest_truth_delay_distance(old, truth_peaks),
                "truth_distance_new": float("nan"),
                "local_error_reduction": reduction,
                "beneficial": reduction > 0.0,
            }
        )
    for new_index, new in enumerate(new_peaks):
        if new_index in new_matched:
            continue
        mask = delay_window_mask(new.coordinate[0])
        reduction = local_error_reduction(base, changed, truth, mask)
        events.append(
            {
                **common(),
                "position_action": "新增",
                "mass_action": "增强",
                "old_h": -1,
                "old_v": -1,
                "old_delay": -1,
                "new_h": -1,
                "new_v": -1,
                "new_delay": new.coordinate[0],
                "shift_h": float("nan"),
                "shift_v": float("nan"),
                "shift_delay": float("nan"),
                "match_cost": float("nan"),
                "old_peak_value": 0.0,
                "new_peak_value": new.value,
                "old_mass": 0.0,
                "new_mass": new.mass,
                "mass_delta": new.mass,
                "truth_distance_old": float("nan"),
                "truth_distance_new": nearest_truth_delay_distance(new, truth_peaks),
                "local_error_reduction": reduction,
                "beneficial": reduction > 0.0,
            }
        )
    return events


def summarize_events(
    events: list[dict[str, object]], count: int
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    per_point = {
        name: np.zeros(count, dtype=np.float64)
        for name in (
            "angle_new_mass",
            "angle_removed_mass",
            "angle_moved_mass",
            "angle_strengthened_mass",
            "angle_weakened_mass",
            "delay_new_mass",
            "delay_removed_mass",
            "delay_moved_mass",
            "delay_strengthened_mass",
            "delay_weakened_mass",
        )
    }
    summary: dict[str, object] = {}
    for domain, prefix in (("角度", "angle"), ("时延", "delay")):
        selected = [event for event in events if event["domain"] == domain]
        position = Counter(str(event["position_action"]) for event in selected)
        mass = Counter(str(event["mass_action"]) for event in selected)
        beneficial = defaultdict(list)
        for event in selected:
            beneficial[str(event["position_action"])].append(bool(event["beneficial"]))
            row = int(event["point_row"])
            action = str(event["position_action"])
            mass_name = str(event["mass_action"])
            old_mass = float(event["old_mass"])
            new_mass = float(event["new_mass"])
            if action == "新增":
                per_point[f"{prefix}_new_mass"][row] += new_mass
            elif action == "消失":
                per_point[f"{prefix}_removed_mass"][row] += old_mass
            elif action == "移动":
                per_point[f"{prefix}_moved_mass"][row] += min(old_mass, new_mass)
            if mass_name == "增强":
                per_point[f"{prefix}_strengthened_mass"][row] += max(
                    0.0, new_mass - old_mass
                )
            elif mass_name == "减弱":
                per_point[f"{prefix}_weakened_mass"][row] += max(
                    0.0, old_mass - new_mass
                )
        summary[prefix] = {
            "event_count": len(selected),
            "position_action_count": dict(position),
            "mass_action_count": dict(mass),
            "beneficial_fraction_by_position": {
                key: float(np.mean(value)) for key, value in beneficial.items()
            },
        }
    return summary, per_point


def compact_anchor_maps() -> tuple[np.ndarray, np.ndarray]:
    pas, pdp = e138.compact_cached_endpoints(ANCHOR_PAS, ANCHOR_PDP)
    angle = np.transpose(pas.sum(axis=1), (0, 3, 1, 2))
    return normalize(angle, (2, 3)), normalize(pdp, 3)


def carrier_uncertainty(
    device: torch.device, batch_size: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    bank = np.load(CARRIER_BANK, mmap_mode="r")
    angle_all = np.empty(
        (len(bank), bank.shape[1], e138.N_UE, e138.MH, e138.MV),
        dtype=np.float32,
    )
    delay_all = np.empty(
        (len(bank), bank.shape[1], e138.MP, e138.N_UE, e138.S_SC),
        dtype=np.float32,
    )
    log_power_all = np.empty((len(bank), bank.shape[1]), dtype=np.float32)
    for index in range(len(bank)):
        angle, delay, log_power = channel_group_maps(
            bank[index], device, batch_size
        )
        angle_all[index] = angle
        delay_all[index] = delay
        log_power_all[index] = log_power
        print(f"E153载体谱 {index + 1}/{len(bank)}", flush=True)
    consensus_angle = normalize(angle_all.mean(axis=0), (2, 3))
    consensus_delay = normalize(delay_all.mean(axis=0), 3)
    angle_dispersion = hellinger_squared(
        angle_all, consensus_angle[None], (3, 4)
    ).mean(axis=(0, 2))
    delay_dispersion = hellinger_squared(
        delay_all, consensus_delay[None], (4,)
    ).mean(axis=(0, 2, 3))
    return (
        {
            "carrier_angle_dispersion": angle_dispersion,
            "carrier_delay_dispersion": delay_dispersion,
            "carrier_log_power_std": log_power_all.std(axis=0, dtype=np.float64),
        },
        {
            "angle": consensus_angle,
            "delay": consensus_delay,
            "log_power": log_power_all.mean(axis=0),
        },
    )


def neighbor_features(
    query_ids: np.ndarray,
    positions: np.ndarray,
    train_angle: np.ndarray,
    train_delay: np.ndarray,
    v7_angle: np.ndarray,
    v7_delay: np.ndarray,
    v12_angle: np.ndarray,
    v12_delay: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    cache = np.load(NEIGHBOR_CACHE, allow_pickle=False)
    if not np.array_equal(cache["query_ids"].astype(np.int64), query_ids):
        raise RuntimeError("E153 neighbor query IDs do not match canonical 500")
    neighbor_ids = cache["neighbor_ids"].astype(np.int64)
    count = len(query_ids)
    consensus_angle = np.empty_like(v7_angle)
    consensus_delay = np.empty_like(v7_delay)
    features = {
        name: np.empty(count, dtype=np.float64)
        for name in (
            "nearest_distance",
            "mean_distance8",
            "mean_distance16",
            "effective_neighbor_count",
            "neighbor_angle_dispersion",
            "neighbor_delay_dispersion",
            "neighbor_angle_alignment",
            "neighbor_delay_alignment",
            "neighbor_angle_distance_to_v7",
            "neighbor_delay_distance_to_v7",
        )
    }
    for row in range(count):
        ids = neighbor_ids[row]
        distances = np.linalg.norm(
            positions[ids, :2] - positions[query_ids[row], None, :2], axis=1
        )
        weight = 1.0 / np.maximum(distances, 1e-3) ** 2
        weight = weight / max(float(weight.sum()), TINY)
        consensus_angle[row] = normalize(
            np.sum(train_angle[ids] * weight[:, None, None, None], axis=0),
            (1, 2),
        )
        consensus_delay[row] = normalize(
            np.sum(train_delay[ids] * weight[:, None, None, None], axis=0), 2
        )
        features["nearest_distance"][row] = float(distances[0])
        features["mean_distance8"][row] = float(np.mean(distances[:8]))
        features["mean_distance16"][row] = float(np.mean(distances[:16]))
        features["effective_neighbor_count"][row] = float(
            1.0 / np.sum(weight * weight)
        )
        features["neighbor_angle_dispersion"][row] = float(
            np.sum(
                weight
                * hellinger_squared(
                    train_angle[ids], consensus_angle[row][None], (2, 3)
                ).mean(axis=1)
            )
        )
        features["neighbor_delay_dispersion"][row] = float(
            np.sum(
                weight
                * hellinger_squared(
                    train_delay[ids], consensus_delay[row][None], (3,)
                ).mean(axis=(1, 2))
            )
        )
    features["neighbor_angle_alignment"] = direction_cosine(
        np.sqrt(consensus_angle) - np.sqrt(v7_angle),
        np.sqrt(v12_angle) - np.sqrt(v7_angle),
    )
    features["neighbor_delay_alignment"] = direction_cosine(
        np.sqrt(consensus_delay) - np.sqrt(v7_delay),
        np.sqrt(v12_delay) - np.sqrt(v7_delay),
    )
    features["neighbor_angle_distance_to_v7"] = hellinger_squared(
        consensus_angle, v7_angle, (2, 3)
    ).mean(axis=1)
    features["neighbor_delay_distance_to_v7"] = hellinger_squared(
        consensus_delay, v7_delay, (3,)
    ).mean(axis=(1, 2))
    return features, consensus_angle, consensus_delay


def point_spectral_features(
    angle: np.ndarray, delay: np.ndarray, log_power: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "v7_angle_entropy": entropy(angle, (2, 3)).mean(axis=1),
        "v7_angle_top1": top_mass(angle, (2, 3), 1).mean(axis=1),
        "v7_angle_top4": top_mass(angle, (2, 3), 4).mean(axis=1),
        "v7_delay_entropy": entropy(delay, (3,)).mean(axis=(1, 2)),
        "v7_delay_top1": top_mass(delay, (3,), 1).mean(axis=(1, 2)),
        "v7_delay_top4": top_mass(delay, (3,), 4).mean(axis=(1, 2)),
        "v7_log_power": log_power.astype(np.float64),
    }


def feature_correlations(
    feature_groups: dict[str, dict[str, np.ndarray]],
    targets: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, features in feature_groups.items():
        for feature_name, feature in features.items():
            for target_name, target in targets.items():
                valid = np.isfinite(feature) & np.isfinite(target)
                if np.sum(valid) < 20 or np.std(feature[valid]) <= TINY:
                    rho, pvalue = 0.0, 1.0
                else:
                    result = spearmanr(feature[valid], target[valid])
                    rho, pvalue = float(result.statistic), float(result.pvalue)
                    if not np.isfinite(rho):
                        rho, pvalue = 0.0, 1.0
                rows.append(
                    {
                        "group": group,
                        "feature": feature_name,
                        "target": target_name,
                        "spearman": rho,
                        "pvalue": pvalue,
                    }
                )
    return rows


def cross_region_predictability(
    matrices: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    fold_id: np.ndarray,
    trees: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, matrix in matrices.items():
        value = np.asarray(matrix, dtype=np.float32)
        for target_name, target in targets.items():
            prediction = np.empty(len(target), dtype=np.float64)
            for outer in range(5):
                train_rows = np.flatnonzero(fold_id != outer)
                held_rows = np.flatnonzero(fold_id == outer)
                model = ExtraTreesRegressor(
                    n_estimators=trees,
                    max_depth=8,
                    min_samples_leaf=10,
                    max_features=0.80,
                    n_jobs=-1,
                    random_state=(seed + outer * 1009) % (2**32),
                )
                model.fit(value[train_rows], target[train_rows])
                prediction[held_rows] = model.predict(value[held_rows])
            rho = float(spearmanr(prediction, target).statistic)
            if not np.isfinite(rho):
                rho = 0.0
            rows.append(
                {
                    "group": group,
                    "target": target_name,
                    "r2": float(r2_score(target, prediction)),
                    "spearman": rho,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_contribution_figure(
    v7: dict[str, float | int], v12: dict[str, float | int]
) -> Path:
    labels = ["PAS", "PDP", "NMSE得分"]
    values = [
        0.4 * (float(v12["pas"]) - float(v7["pas"])),
        0.4 * (float(v12["pdp"]) - float(v7["pdp"])),
        0.2 * (float(v12["nmse_score"]) - float(v7["nmse_score"])),
    ]
    path = VISUAL_DIR / "01_V12相对V7的总分提升来源.png"
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(labels, values, color=["#2563eb", "#16a34a", "#d97706"])
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_ylabel("对官方总分的贡献")
    ax.set_title("V12相对V7的总分提升主要来自PAS")
    total = sum(values)
    for bar, value in zip(bars, values, strict=True):
        share = value / total if abs(total) > TINY else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:+.6f}\n{share:.1%}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def spatial_gain_figure(
    positions: np.ndarray,
    query_ids: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    pas_gain: np.ndarray,
    pdp_gain: np.ndarray,
) -> Path:
    path = VISUAL_DIR / "02_500点PAS与PDP增益空间分布.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    limit = max(
        float(np.quantile(np.abs(pas_gain), 0.98)),
        float(np.quantile(np.abs(pdp_gain), 0.98)),
        1e-4,
    )
    for ax, value, title in zip(
        axes,
        (pas_gain, pdp_gain),
        ("逐点PAS得分变化", "逐点PDP得分变化"),
        strict=True,
    ):
        scatter = ax.scatter(
            positions[query_ids, 0],
            positions[query_ids, 1],
            c=value,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            s=28,
            linewidths=0.2,
            edgecolors="#374151",
        )
        for site_value in np.unique(site):
            bs = bs_positions[int(site_value)]
            ax.scatter(
                bs[0], bs[1], marker="*", s=180, color="#111827", zorder=5
            )
            ax.text(bs[0], bs[1], f"  基站{int(site_value)}", va="center")
        ax.set_title(title)
        ax.set_xlabel("X坐标 / 米")
        ax.set_ylabel("Y坐标 / 米")
        ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(scatter, ax=ax, shrink=0.82)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def event_summary_figure(
    event_summary: dict[str, object], events: list[dict[str, object]]
) -> Path:
    path = VISUAL_DIR / "03_角度与时延峰动作统计.png"
    position_order = ["保留", "移动", "新增", "消失"]
    mass_order = ["增强", "减弱", "基本不变"]
    colors = {
        "保留": "#64748b",
        "移动": "#7c3aed",
        "新增": "#16a34a",
        "消失": "#dc2626",
        "增强": "#2563eb",
        "减弱": "#f97316",
        "基本不变": "#94a3b8",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for column, (domain, prefix) in enumerate((("角度", "angle"), ("时延", "delay"))):
        selected = [event for event in events if event["domain"] == domain]
        position = Counter(str(event["position_action"]) for event in selected)
        mass = Counter(str(event["mass_action"]) for event in selected)
        axes[0, column].bar(
            position_order,
            [position[name] for name in position_order],
            color=[colors[name] for name in position_order],
        )
        axes[0, column].set_title(f"{domain}峰的位置动作")
        axes[0, column].set_ylabel("事件数量")
        for tick in axes[0, column].get_xticklabels():
            tick.set_rotation(15)
        axes[1, column].bar(
            mass_order,
            [mass[name] for name in mass_order],
            color=[colors[name] for name in mass_order],
        )
        axes[1, column].set_title(f"{domain}峰的质量动作")
        axes[1, column].set_ylabel("事件数量")
        beneficial = event_summary[prefix]["beneficial_fraction_by_position"]
        text = "  ".join(
            f"{name}正确率 {float(beneficial.get(name, 0.0)):.1%}"
            for name in position_order
            if name in beneficial
        )
        axes[0, column].text(
            0.02,
            0.98,
            text,
            transform=axes[0, column].transAxes,
            va="top",
            fontsize=9,
        )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def concentration_figure(
    v7_angle: np.ndarray,
    v12_angle: np.ndarray,
    truth_angle: np.ndarray,
    v7_delay: np.ndarray,
    v12_delay: np.ndarray,
    truth_delay: np.ndarray,
) -> Path:
    path = VISUAL_DIR / "04_V7与V12谱集中度对比.png"
    metrics = {
        "角度Top1质量": (
            top_mass(v7_angle, (2, 3), 1).mean(1),
            top_mass(v12_angle, (2, 3), 1).mean(1),
            top_mass(truth_angle, (2, 3), 1).mean(1),
        ),
        "角度Top4质量": (
            top_mass(v7_angle, (2, 3), 4).mean(1),
            top_mass(v12_angle, (2, 3), 4).mean(1),
            top_mass(truth_angle, (2, 3), 4).mean(1),
        ),
        "时延Top1质量": (
            top_mass(v7_delay, (3,), 1).mean((1, 2)),
            top_mass(v12_delay, (3,), 1).mean((1, 2)),
            top_mass(truth_delay, (3,), 1).mean((1, 2)),
        ),
        "时延Top4质量": (
            top_mass(v7_delay, (3,), 4).mean((1, 2)),
            top_mass(v12_delay, (3,), 4).mean((1, 2)),
            top_mass(truth_delay, (3,), 4).mean((1, 2)),
        ),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (title, values) in zip(axes.flat, metrics.items(), strict=True):
        data = [value[np.isfinite(value)] for value in values]
        box = ax.boxplot(
            data, tick_labels=["V7", "V12", "真值"], patch_artist=True
        )
        for patch, color in zip(
            box["boxes"], ["#93c5fd", "#86efac", "#fca5a5"], strict=True
        ):
            patch.set_facecolor(color)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def binned_relation_figure(
    feature_groups: dict[str, dict[str, np.ndarray]],
    targets: dict[str, np.ndarray],
) -> Path:
    path = VISUAL_DIR / "05_V7不确定度与修正幅度关系.png"
    pairs = [
        ("17载体角度分歧", feature_groups["V7自身"]["17载体角度分歧"], targets["angle_action"]),
        ("17载体时延分歧", feature_groups["V7自身"]["17载体时延分歧"], targets["delay_action"]),
        ("邻居角度分歧", feature_groups["邻居信道"]["邻居角度分歧"], targets["angle_action"]),
        ("邻居时延分歧", feature_groups["邻居信道"]["邻居时延分歧"], targets["delay_action"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (name, feature, target) in zip(axes.flat, pairs, strict=True):
        edges = np.quantile(feature, np.linspace(0.0, 1.0, 6))
        edges = np.maximum.accumulate(edges)
        x, y, error = [], [], []
        for index in range(5):
            if index == 4:
                mask = (feature >= edges[index]) & (feature <= edges[index + 1])
            else:
                mask = (feature >= edges[index]) & (feature < edges[index + 1])
            if not np.any(mask):
                continue
            x.append(float(np.mean(feature[mask])))
            y.append(float(np.mean(target[mask])))
            error.append(float(np.std(target[mask]) / math.sqrt(np.sum(mask))))
        ax.errorbar(x, y, yerr=error, marker="o", color="#2563eb", capsize=3)
        rho = float(spearmanr(feature, target).statistic)
        ax.set_title(f"{name}，Spearman={rho:+.3f}")
        ax.set_xlabel(name)
        ax.set_ylabel("对应修正幅度")
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def neighbor_alignment_figure(
    neighbor_features_value: dict[str, np.ndarray],
    pas_gain: np.ndarray,
    pdp_gain: np.ndarray,
) -> Path:
    path = VISUAL_DIR / "06_邻居修正方向与实际增益关系.png"
    pairs = [
        (
            "邻居角度方向一致性",
            neighbor_features_value["neighbor_angle_alignment"],
            pas_gain,
        ),
        (
            "邻居时延方向一致性",
            neighbor_features_value["neighbor_delay_alignment"],
            pdp_gain,
        ),
        (
            "邻居角度分歧",
            neighbor_features_value["neighbor_angle_dispersion"],
            pas_gain,
        ),
        (
            "邻居时延分歧",
            neighbor_features_value["neighbor_delay_dispersion"],
            pdp_gain,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (name, feature, gain) in zip(axes.flat, pairs, strict=True):
        ax.scatter(feature, gain, s=14, alpha=0.45, color="#0f766e")
        rho = float(spearmanr(feature, gain).statistic)
        ax.axhline(0.0, color="#111827", linewidth=0.8)
        ax.set_title(f"{name}，Spearman={rho:+.3f}")
        ax.set_xlabel(name)
        ax.set_ylabel("对应官方谱得分增益")
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def map_correlation_figure(correlations: list[dict[str, object]]) -> Path:
    path = VISUAL_DIR / "07_地图结构与修正动作相关性.png"
    selected = [row for row in correlations if row["group"] == "地图结构"]
    targets = list(TARGET_LABELS)
    feature_score: dict[str, float] = defaultdict(float)
    lookup = {}
    for row in selected:
        lookup[(str(row["feature"]), str(row["target"]))] = float(row["spearman"])
        feature_score[str(row["feature"])] = max(
            feature_score[str(row["feature"])], abs(float(row["spearman"]))
        )
    features = sorted(feature_score, key=feature_score.get, reverse=True)[:12]
    matrix = np.asarray(
        [[lookup.get((feature, target), 0.0) for target in targets] for feature in features]
    )
    fig, ax = plt.subplots(figsize=(11, 7.5))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(np.arange(len(targets)), [TARGET_LABELS[name] for name in targets])
    ax.set_yticks(np.arange(len(features)), features)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("地图结构特征与V7→V12修正动作的Spearman相关")
    fig.colorbar(image, ax=ax, label="Spearman相关系数")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def predictability_figure(rows: list[dict[str, object]]) -> Path:
    path = VISUAL_DIR / "08_不同信息组的跨空间区域解释力.png"
    groups = ["V7自身", "邻居信道", "位置", "地图结构", "全部信息"]
    targets = ["angle_action", "delay_action", "pas_gain", "pdp_gain"]
    lookup = {
        (str(row["group"]), str(row["target"])): float(row["spearman"])
        for row in rows
    }
    matrix = np.asarray(
        [[lookup[(group, target)] for target in targets] for group in groups]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.4, vmax=0.8, aspect="auto")
    ax.set_xticks(np.arange(len(targets)), [TARGET_LABELS[name] for name in targets])
    ax.set_yticks(np.arange(len(groups)), groups)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:+.3f}", ha="center", va="center")
    ax.set_title("固定模型跨空间区域预测标量修正目标的Spearman")
    fig.colorbar(image, ax=ax, label="预测值与真实值的Spearman")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def regime_figure(
    regime: np.ndarray,
    angle_action: np.ndarray,
    delay_action: np.ndarray,
    pas_gain: np.ndarray,
    pdp_gain: np.ndarray,
) -> Path:
    path = VISUAL_DIR / "09_不同传播难度区域的修正幅度与收益.png"
    x = np.arange(len(REGIME_NAMES))
    values = []
    for target in (angle_action, delay_action, pas_gain, pdp_gain):
        values.append(
            [float(np.mean(target[regime == index])) for index in range(len(REGIME_NAMES))]
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, values[0], width, label="角度修正", color="#2563eb")
    axes[0].bar(x + width / 2, values[1], width, label="时延修正", color="#16a34a")
    axes[0].set_xticks(x, ["清晰", "轻度", "中度", "重度"])
    axes[0].set_title("平均修正幅度")
    axes[0].legend()
    axes[1].bar(x - width / 2, values[2], width, label="PAS增益", color="#7c3aed")
    axes[1].bar(x + width / 2, values[3], width, label="PDP增益", color="#d97706")
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_xticks(x, ["清晰", "轻度", "中度", "重度"])
    axes[1].set_title("平均官方谱得分增益")
    axes[1].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def event_action_details(events: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for domain, prefix in (("角度", "angle"), ("时延", "delay")):
        selected = [event for event in events if event["domain"] == domain]
        position_rows = []
        for action in ("保留", "移动", "新增", "消失"):
            rows = [event for event in selected if event["position_action"] == action]
            if not rows:
                continue
            position_rows.append(
                {
                    "action": action,
                    "count": len(rows),
                    "fraction": len(rows) / len(selected),
                    "beneficial_fraction": float(
                        np.mean([bool(row["beneficial"]) for row in rows])
                    ),
                    "mean_local_error_reduction": float(
                        np.mean([float(row["local_error_reduction"]) for row in rows])
                    ),
                }
            )
        mass_rows = []
        for action in ("增强", "减弱", "基本不变"):
            rows = [event for event in selected if event["mass_action"] == action]
            if not rows:
                continue
            mass_rows.append(
                {
                    "action": action,
                    "count": len(rows),
                    "fraction": len(rows) / len(selected),
                    "beneficial_fraction": float(
                        np.mean([bool(row["beneficial"]) for row in rows])
                    ),
                    "mean_local_error_reduction": float(
                        np.mean([float(row["local_error_reduction"]) for row in rows])
                    ),
                }
            )

        top_bins: dict[str, list[dict[str, object]]] = {}
        specifications = (
            ("增强", "mass_action", "增强", "new", "mass_delta"),
            ("减弱", "mass_action", "减弱", "old", "mass_delta"),
            ("移动", "position_action", "移动", "new", "moved"),
            ("新增", "position_action", "新增", "new", "new_mass"),
            ("消失", "position_action", "消失", "old", "old_mass"),
        )
        for label, field, expected, coordinate_side, mass_field in specifications:
            aggregate: dict[tuple[int, ...], dict[str, object]] = defaultdict(
                lambda: {"count": 0, "mass": 0.0, "beneficial": []}
            )
            for event in selected:
                if event[field] != expected:
                    continue
                if domain == "角度":
                    coordinate = (
                        int(event[f"{coordinate_side}_h"]),
                        int(event[f"{coordinate_side}_v"]),
                    )
                else:
                    coordinate = (int(event[f"{coordinate_side}_delay"]),)
                if mass_field == "moved":
                    mass = min(float(event["old_mass"]), float(event["new_mass"]))
                elif mass_field == "mass_delta":
                    mass = abs(float(event["mass_delta"]))
                else:
                    mass = float(event[mass_field])
                aggregate[coordinate]["count"] = int(aggregate[coordinate]["count"]) + 1
                aggregate[coordinate]["mass"] = float(aggregate[coordinate]["mass"]) + mass
                aggregate[coordinate]["beneficial"].append(bool(event["beneficial"]))
            ranked = sorted(
                aggregate.items(), key=lambda item: float(item[1]["mass"]), reverse=True
            )[:12]
            top_bins[label] = [
                {
                    "coordinate": list(coordinate),
                    "count": int(value["count"]),
                    "mass": float(value["mass"]),
                    "beneficial_fraction": float(np.mean(value["beneficial"])),
                }
                for coordinate, value in ranked
            ]

        shift_rows = []
        moved = [event for event in selected if event["position_action"] == "移动"]
        if domain == "角度":
            keys = Counter(
                (float(event["shift_h"]), float(event["shift_v"])) for event in moved
            )
            for (shift_h, shift_v), count in sorted(
                keys.items(), key=lambda item: item[1], reverse=True
            ):
                rows = [
                    event
                    for event in moved
                    if float(event["shift_h"]) == shift_h
                    and float(event["shift_v"]) == shift_v
                ]
                shift_rows.append(
                    {
                        "shift_h": shift_h,
                        "shift_v": shift_v,
                        "count": count,
                        "beneficial_fraction": float(
                            np.mean([bool(row["beneficial"]) for row in rows])
                        ),
                    }
                )
        else:
            keys = Counter(float(event["shift_delay"]) for event in moved)
            for shift, count in sorted(keys.items()):
                rows = [event for event in moved if float(event["shift_delay"]) == shift]
                shift_rows.append(
                    {
                        "shift_delay": shift,
                        "count": count,
                        "beneficial_fraction": float(
                            np.mean([bool(row["beneficial"]) for row in rows])
                        ),
                    }
                )
        site_rows = []
        for site_value in (0, 1):
            for action in ("保留", "移动", "新增", "消失"):
                rows = [
                    event
                    for event in selected
                    if int(event["site"]) == site_value
                    and event["position_action"] == action
                ]
                if not rows:
                    continue
                site_rows.append(
                    {
                        "site": site_value,
                        "action": action,
                        "count": len(rows),
                        "beneficial_fraction": float(
                            np.mean([bool(row["beneficial"]) for row in rows])
                        ),
                        "mean_local_error_reduction": float(
                            np.mean(
                                [float(row["local_error_reduction"]) for row in rows]
                            )
                        ),
                    }
                )
        output[prefix] = {
            "position_actions": position_rows,
            "mass_actions": mass_rows,
            "top_bins_by_mass": top_bins,
            "move_distribution": shift_rows,
            "position_actions_by_site": site_rows,
        }
    return output


def angle_action_location_figure(events: list[dict[str, object]]) -> Path:
    path = VISUAL_DIR / "12_角度峰动作发生位置.png"
    selected = [event for event in events if event["domain"] == "角度"]
    specifications = (
        ("增强", "mass_action", "增强", "new", "mass_delta"),
        ("减弱", "mass_action", "减弱", "old", "mass_delta"),
        ("移动后的峰", "position_action", "移动", "new", "moved"),
        ("新增", "position_action", "新增", "new", "new_mass"),
        ("消失", "position_action", "消失", "old", "old_mass"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, (label, field, expected, side, mass_field) in zip(
        axes.flat, specifications, strict=False
    ):
        value = np.zeros((e138.MH, e138.MV), dtype=np.float64)
        rows = [event for event in selected if event[field] == expected]
        for event in rows:
            h = int(event[f"{side}_h"])
            v = int(event[f"{side}_v"])
            if mass_field == "moved":
                mass = min(float(event["old_mass"]), float(event["new_mass"]))
            elif mass_field == "mass_delta":
                mass = abs(float(event["mass_delta"]))
            else:
                mass = float(event[mass_field])
            value[h, v] += mass
        image = ax.imshow(value.T, origin="lower", aspect="auto", cmap="magma")
        beneficial = float(np.mean([bool(row["beneficial"]) for row in rows]))
        ax.set_title(f"{label}：质量{value.sum():.1f}，改善率{beneficial:.1%}")
        ax.set_xlabel("水平角度格")
        ax.set_ylabel("垂直角度格")
        fig.colorbar(image, ax=ax, shrink=0.80)
    axes.flat[-1].axis("off")
    fig.suptitle("V7→V12角度峰动作集中在哪些角度格")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def delay_action_location_figure(events: list[dict[str, object]]) -> Path:
    path = VISUAL_DIR / "13_时延峰动作发生位置.png"
    selected = [event for event in events if event["domain"] == "时延"]
    specifications = (
        ("增强", "mass_action", "增强", "new", "mass_delta"),
        ("减弱", "mass_action", "减弱", "old", "mass_delta"),
        ("新增", "position_action", "新增", "new", "new_mass"),
        ("消失", "position_action", "消失", "old", "old_mass"),
    )
    display_bins = 32
    labels = [str(index) for index in range(display_bins)] + ["32-191"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
    for ax, (label, field, expected, side, mass_field) in zip(
        axes.flat, specifications, strict=True
    ):
        full = np.zeros(e138.S_SC, dtype=np.float64)
        rows = [event for event in selected if event[field] == expected]
        for event in rows:
            delay = int(event[f"{side}_delay"])
            if mass_field == "mass_delta":
                mass = abs(float(event["mass_delta"]))
            else:
                mass = float(event[mass_field])
            full[delay] += mass
        value = np.concatenate((full[:display_bins], [full[display_bins:].sum()]))
        ax.bar(np.arange(len(value)), value, color="#2563eb")
        ax.set_xticks(np.arange(0, len(value), 4), labels[::4], rotation=30)
        beneficial = float(np.mean([bool(row["beneficial"]) for row in rows]))
        ax.set_title(f"{label}：质量{full.sum():.1f}，改善率{beneficial:.1%}")
        ax.set_xlabel("时延格")
        ax.set_ylabel("累计动作质量")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("V7→V12时延峰动作集中在哪些时延格")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def quartile_relation(
    feature: np.ndarray,
    target: np.ndarray,
    gain: np.ndarray,
) -> list[dict[str, float | int]]:
    edges = np.quantile(feature, np.linspace(0.0, 1.0, 5))
    rows: list[dict[str, float | int]] = []
    for index in range(4):
        if index == 3:
            mask = (feature >= edges[index]) & (feature <= edges[index + 1])
        else:
            mask = (feature >= edges[index]) & (feature < edges[index + 1])
        rows.append(
            {
                "quartile": index + 1,
                "minimum": float(edges[index]),
                "maximum": float(edges[index + 1]),
                "count": int(np.sum(mask)),
                "target_mean": float(np.mean(target[mask])),
                "gain_mean": float(np.mean(gain[mask])),
            }
        )
    return rows


def point_strata_summary(
    site: np.ndarray,
    regime: np.ndarray,
    angle_action: np.ndarray,
    delay_action: np.ndarray,
    pas_gain: np.ndarray,
    pdp_gain: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {"site": [], "regime": []}
    for site_value in (0, 1):
        mask = site == site_value
        output["site"].append(
            {
                "site": site_value,
                "count": int(np.sum(mask)),
                "angle_action_mean": float(np.mean(angle_action[mask])),
                "delay_action_mean": float(np.mean(delay_action[mask])),
                "pas_gain_mean": float(np.mean(pas_gain[mask])),
                "pdp_gain_mean": float(np.mean(pdp_gain[mask])),
                "pas_positive_fraction": float(np.mean(pas_gain[mask] > 0.0)),
                "pdp_positive_fraction": float(np.mean(pdp_gain[mask] > 0.0)),
            }
        )
    for regime_value, name in enumerate(REGIME_NAMES):
        mask = regime == regime_value
        output["regime"].append(
            {
                "regime": name,
                "count": int(np.sum(mask)),
                "angle_action_mean": float(np.mean(angle_action[mask])),
                "delay_action_mean": float(np.mean(delay_action[mask])),
                "pas_gain_mean": float(np.mean(pas_gain[mask])),
                "pdp_gain_mean": float(np.mean(pdp_gain[mask])),
                "pas_positive_fraction": float(np.mean(pas_gain[mask] > 0.0)),
                "pdp_positive_fraction": float(np.mean(pdp_gain[mask] > 0.0)),
            }
        )
    return output


def information_strata_figure(
    v7_named: dict[str, np.ndarray],
    neighbor_named: dict[str, np.ndarray],
    map_named: dict[str, np.ndarray],
    angle_action: np.ndarray,
    delay_action: np.ndarray,
    pas_gain: np.ndarray,
    pdp_gain: np.ndarray,
    event_point: dict[str, np.ndarray],
) -> Path:
    path = VISUAL_DIR / "14_不确定度邻居与地图结构分层现象.png"
    panels = (
        (
            "V7角度Top1质量",
            v7_named["V7角度Top1质量"],
            angle_action,
            pas_gain,
            "角度修正幅度",
            "PAS增益",
        ),
        (
            "邻居角度分歧",
            neighbor_named["邻居角度分歧"],
            angle_action,
            pas_gain,
            "角度修正幅度",
            "PAS增益",
        ),
        (
            "V7时延Top1质量",
            v7_named["V7时延Top1质量"],
            delay_action,
            pdp_gain,
            "时延修正幅度",
            "PDP增益",
        ),
        (
            "扇形射线平均遮挡数",
            map_named["扇形射线平均遮挡数"],
            event_point["angle_new_mass"],
            pas_gain,
            "角度新增峰质量",
            "PAS增益",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = np.arange(1, 5)
    for ax, (title, feature, target, gain, target_label, gain_label) in zip(
        axes.flat, panels, strict=True
    ):
        rows = quartile_relation(feature, target, gain)
        left = np.asarray([float(row["target_mean"]) for row in rows])
        right = np.asarray([float(row["gain_mean"]) for row in rows])
        ax.plot(x, left, marker="o", color="#2563eb", label=target_label)
        other = ax.twinx()
        other.plot(x, right, marker="s", color="#dc2626", label=gain_label)
        ax.set_xticks(x, ["低", "中低", "中高", "高"])
        ax.set_xlabel(f"{title}四分位")
        ax.set_ylabel(target_label, color="#2563eb")
        other.set_ylabel(gain_label, color="#dc2626")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        handles = ax.get_lines() + other.get_lines()
        ax.legend(handles, [item.get_label() for item in handles], loc="best")
    fig.suptitle("信息能否说明V12在哪里修、修后是否真正得分")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def angle_examples_figure(
    query_ids: np.ndarray,
    v7_angle: np.ndarray,
    v12_angle: np.ndarray,
    truth_angle: np.ndarray,
    pas_gain: np.ndarray,
) -> Path:
    positive = np.argsort(-pas_gain)[:3]
    negative = np.asarray([int(np.argmin(pas_gain))])
    rows = np.concatenate((positive, negative))
    path = VISUAL_DIR / "10_典型角度峰修正示例.png"
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.1 * len(rows)), constrained_layout=True)
    for output_row, point_row in enumerate(rows):
        group_gain = []
        for ue in range(e138.N_UE):
            before = np.dot(v7_angle[point_row, ue].ravel(), truth_angle[point_row, ue].ravel())
            after = np.dot(v12_angle[point_row, ue].ravel(), truth_angle[point_row, ue].ravel())
            group_gain.append(after - before)
        ue = int(np.argmax(np.abs(group_gain)))
        maps = (
            v7_angle[point_row, ue],
            v12_angle[point_row, ue],
            truth_angle[point_row, ue],
            v12_angle[point_row, ue] - v7_angle[point_row, ue],
        )
        maximum = max(float(np.max(value)) for value in maps[:3])
        delta_limit = max(float(np.max(np.abs(maps[3]))), 1e-4)
        for column, value in enumerate(maps):
            cmap = "RdBu_r" if column == 3 else "viridis"
            vmin = -delta_limit if column == 3 else 0.0
            vmax = delta_limit if column == 3 else maximum
            image = axes[output_row, column].imshow(
                value.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
            )
            axes[output_row, column].set_xlabel("水平角度格")
            axes[output_row, column].set_ylabel("垂直角度格")
            fig.colorbar(image, ax=axes[output_row, column], shrink=0.75)
        axes[output_row, 0].set_title(
            f"点{int(query_ids[point_row])} UE{ue} V7\nPAS增益 {pas_gain[point_row]:+.4f}"
        )
        axes[output_row, 1].set_title("V12")
        axes[output_row, 2].set_title("官方真值")
        axes[output_row, 3].set_title("V12 - V7")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def delay_examples_figure(
    query_ids: np.ndarray,
    v7_delay: np.ndarray,
    v12_delay: np.ndarray,
    truth_delay: np.ndarray,
    pdp_gain: np.ndarray,
) -> Path:
    positive = np.argsort(-pdp_gain)[:3]
    negative = np.asarray([int(np.argmin(pdp_gain))])
    rows = np.concatenate((positive, negative))
    path = VISUAL_DIR / "11_典型时延峰修正示例.png"
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 3.0 * len(rows)), constrained_layout=True)
    for output_row, point_row in enumerate(rows):
        group_gain = np.empty((e138.MP, e138.N_UE), dtype=np.float64)
        for polarization in range(e138.MP):
            for ue in range(e138.N_UE):
                truth = truth_delay[point_row, polarization, ue]
                base = v7_delay[point_row, polarization, ue]
                changed = v12_delay[point_row, polarization, ue]
                group_gain[polarization, ue] = np.dot(changed, truth) - np.dot(base, truth)
        polarization, ue = np.unravel_index(np.argmax(np.abs(group_gain)), group_gain.shape)
        ax = axes[output_row]
        ax.plot(v7_delay[point_row, polarization, ue], label="V7", linewidth=1.6)
        ax.plot(v12_delay[point_row, polarization, ue], label="V12", linewidth=1.6)
        ax.plot(truth_delay[point_row, polarization, ue], label="官方真值", linewidth=1.6)
        ax.set_title(
            f"点{int(query_ids[point_row])} P{polarization}-UE{ue}，PDP增益 {pdp_gain[point_row]:+.4f}"
        )
        ax.set_xlabel("时延格")
        ax.set_ylabel("归一化能量")
        ax.legend(ncol=3)
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_report(
    report: dict[str, object],
    figure_paths: list[Path],
) -> str:
    score = report["score_reproduction"]
    event = report["peak_events"]
    details = report["peak_event_details"]
    strata = report["point_strata"]
    quartiles = report["information_quartiles"]
    findings = report["findings"]
    lines = [
        "# V7与V12峰动作数据分析",
        "",
        "本报告只描述数据现象，不设计修正模型。所有结论使用固定500点、官方PAS/PDP/NMSE口径。",
        "",
        "## 结果数据、图",
        "",
        f"V7完整本地分为`{score['v7']['score']:.6f}`，V12为`{score['v12']['score']:.6f}`，总增益为`{score['delta']:+.6f}`。",
        "",
    ]
    for path in figure_paths:
        relative = Path("../../visualizations/复赛V7与V12峰动作分析") / path.name
        lines.extend((f"### {path.stem}", "", f"![{path.stem}]({relative.as_posix()})", ""))
    lines.extend(
        (
            "## 核心结果",
            "",
            f"PAS贡献总分增益的`{findings['pas_score_share']:.1%}`，PDP贡献`{findings['pdp_score_share']:.1%}`，NMSE得分只贡献`{findings['nmse_score_share']:.1%}`。V12使角度Top1质量中位数由`{findings['angle_top1_v7_median']:.4f}`提高到`{findings['angle_top1_v12_median']:.4f}`，真值为`{findings['angle_top1_truth_median']:.4f}`；时延Top1由`{findings['delay_top1_v7_median']:.4f}`提高到`{findings['delay_top1_v12_median']:.4f}`，真值为`{findings['delay_top1_truth_median']:.4f}`。",
            "",
            f"角度修正方向与真值需求的中位余弦为`{findings['angle_direction_alignment_median']:.4f}`，时延为`{findings['delay_direction_alignment_median']:.4f}`。这说明V12整体方向明显优于随机，但远没有逐点完全修对。",
            "",
            "## 角度峰动作",
            "",
            "| 动作 | 数量 | 占角度事件 | 局部改善率 | 平均局部误差减少 |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for row in details["angle"]["position_actions"]:
        lines.append(
            f"| {row['action']} | {row['count']} | {row['fraction']:.1%} | "
            f"{row['beneficial_fraction']:.1%} | {row['mean_local_error_reduction']:+.4f} |"
        )
    lines.extend(
        (
            "",
            "角度移动几乎都是相邻格移动：水平1格`317`次、垂直1格`303`次，水平2格只有`12`次。移动前最近真值峰距离中位数为`1.0`，移动后为`0.5`。",
            "",
            "按动作质量排序的主要角度格如下。坐标格式为`[水平格, 垂直格]`；改善率只表示该局部窗口相对官方真值是否变好。",
            "",
            "| 动作 | 主要角度格：累计动作质量 / 改善率 |",
            "| --- | --- |",
        )
    )
    for action in ("增强", "减弱", "移动", "新增", "消失"):
        entries = details["angle"]["top_bins_by_mass"][action][:6]
        text_value = "；".join(
            f"{entry['coordinate']}：{entry['mass']:.2f}/{entry['beneficial_fraction']:.1%}"
            for entry in entries
        )
        lines.append(f"| {action} | {text_value} |")

    lines.extend(
        (
            "",
            "## 时延峰动作",
            "",
            "| 动作 | 数量 | 占时延事件 | 局部改善率 | 平均局部误差减少 |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for row in details["delay"]["position_actions"]:
        lines.append(
            f"| {row['action']} | {row['count']} | {row['fraction']:.1%} | "
            f"{row['beneficial_fraction']:.1%} | {row['mean_local_error_reduction']:+.4f} |"
        )
    lines.extend(
        (
            "",
            "`764`个移动峰中，`639`个只移动1个delay格，`107`个移动2格，只有`18`个移动3格。新增峰并没有形成可靠补全：整体改善率只有`10.8%`。",
            "",
            "| 动作 | 主要delay格：累计动作质量 / 改善率 |",
            "| --- | --- |",
        )
    )
    for action in ("增强", "减弱", "移动", "新增", "消失"):
        entries = details["delay"]["top_bins_by_mass"][action][:8]
        text_value = "；".join(
            f"{entry['coordinate'][0]}：{entry['mass']:.2f}/{entry['beneficial_fraction']:.1%}"
            for entry in entries
        )
        lines.append(f"| {action} | {text_value} |")

    lines.extend(
        (
            "",
            "## 基站与传播区域",
            "",
            "| 基站 | 点数 | 角度动作 | delay动作 | PAS增益 | PDP增益 | PAS正向点 | PDP正向点 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for row in strata["site"]:
        lines.append(
            f"| {row['site']} | {row['count']} | {row['angle_action_mean']:.4f} | "
            f"{row['delay_action_mean']:.4f} | {row['pas_gain_mean']:+.4f} | "
            f"{row['pdp_gain_mean']:+.4f} | {row['pas_positive_fraction']:.1%} | "
            f"{row['pdp_positive_fraction']:.1%} |"
        )
    lines.extend(
        (
            "",
            f"{findings['site_asymmetry_conclusion']}",
            "",
            "| 传播区域 | 点数 | 角度动作 | delay动作 | PAS增益 | PDP增益 | PAS正向点 | PDP正向点 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    regime_labels = {"clear": "清晰", "light": "轻度", "medium": "中度", "heavy": "重度"}
    for row in strata["regime"]:
        lines.append(
            f"| {regime_labels[str(row['regime'])]} | {row['count']} | "
            f"{row['angle_action_mean']:.4f} | {row['delay_action_mean']:.4f} | "
            f"{row['pas_gain_mean']:+.4f} | {row['pdp_gain_mean']:+.4f} | "
            f"{row['pas_positive_fraction']:.1%} | {row['pdp_positive_fraction']:.1%} |"
        )

    angle_confidence = quartiles["v7_angle_top1"]
    neighbor_angle = quartiles["neighbor_angle_dispersion"]
    obstruction = quartiles["map_obstruction_angle_new"]
    lines.extend(
        (
            "",
            "## 与三类信息的关系",
            "",
            f"**V7自身不确定度。** V7角度Top1最低四分位的平均角度修正为`{angle_confidence[0]['target_mean']:.4f}`、PAS增益`{angle_confidence[0]['gain_mean']:+.4f}`；最高四分位分别为`{angle_confidence[-1]['target_mean']:.4f}`、`{angle_confidence[-1]['gain_mean']:+.4f}`。V7越分散，V12通常改得越多，且平均收益更高。跨空间区域的最佳Spearman为`+0.375`，但R2只有`0.071`，所以它能判断风险大小，不能还原完整动作。",
            "",
            f"**官方邻居信道。** 邻居角度分歧从最低到最高四分位时，平均修正由`{neighbor_angle[0]['target_mean']:.4f}`升至`{neighbor_angle[-1]['target_mean']:.4f}`，PAS增益由`{neighbor_angle[0]['gain_mean']:+.4f}`升至`{neighbor_angle[-1]['gain_mean']:+.4f}`。但邻居修正方向一致性与PAS/PDP得分增益的Spearman只有`-0.185/-0.086`，不能把邻居平均方向直接当答案。邻居信息主要说明该位置是否处于传播边界。",
            "",
            f"**地图结构。** 扇形遮挡从最低到最高四分位时，V12角度新增峰质量由`{obstruction[0]['target_mean']:.4f}`升至`{obstruction[-1]['target_mean']:.4f}`，PAS增益反而由`{obstruction[0]['gain_mean']:+.4f}`降至`{obstruction[-1]['gain_mean']:+.4f}`。地图与峰出生风险有明显原始相关，但地图组跨空间区域预测PAS增益的最佳Spearman只有`+0.114`。地图能标出困难区域，不能独立给出精确峰位置和能量。",
            "",
            "## 能说明什么问题",
            "",
            f"1. {findings['primary_conclusion']}",
            f"2. {findings['delay_conclusion']}",
            f"3. {findings['uncertainty_conclusion']}",
            f"4. {findings['neighbor_conclusion']}",
            f"5. {findings['map_conclusion']}",
            "",
            "这里的局部改善率基于峰周围窗口的L1误差变化；不同峰窗口可能重叠，因此它用于判断动作可靠性，不能相加解释完整官方总分。本文不据此提出或训练解决方案。",
            "",
            "## 数据文件",
            "",
            f"- 500点明细：`{PATHS.relative(POINT_CSV)}`",
            f"- 峰动作明细：`{PATHS.relative(EVENT_CSV)}`",
            f"- 相关性表：`{PATHS.relative(CORRELATION_CSV)}`",
            f"- 信息组解释力：`{PATHS.relative(PREDICTABILITY_CSV)}`",
            f"- 机器报告：`{PATHS.relative(REPORT)}`",
            f"- 数组缓存：`{PATHS.relative(METRICS)}`",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_chinese_font()

    required = (
        V7_CHANNEL,
        V12_CHANNEL,
        TRAIN_CHANNEL,
        GEOMETRY,
        ANCHOR_PAS,
        ANCHOR_PDP,
        CARRIER_BANK,
        CARRIER_META,
        NEIGHBOR_CACHE,
        REGIME_CACHE,
        OOF_META,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    setup, train_positions, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    geometry_cache = np.load(GEOMETRY, allow_pickle=False)
    geometry = geometry_cache["train"].astype(np.float32)
    geometry_names = geometry_cache["names"].astype(str)
    geometry_query = geometry[query_ids]
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)

    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    v7_channel = np.load(V7_CHANNEL, mmap_mode="r")
    v12_channel = np.load(V12_CHANNEL, mmap_mode="r")

    v7_values = channel_point_metrics(
        v7_channel, truth_all, query_ids, device, args.batch_size
    )
    v12_values = channel_point_metrics(
        v12_channel, truth_all, query_ids, device, args.batch_size
    )
    v7_summary = statistics(*v7_values)
    v12_summary = statistics(*v12_values)
    if abs(float(v7_summary["score"]) - 0.6444563457343335) > 1e-8:
        raise RuntimeError(f"V7 score mismatch: {v7_summary}")
    if abs(float(v12_summary["score"]) - 0.6775208074756943) > 1e-8:
        raise RuntimeError(f"V12 score mismatch: {v12_summary}")
    print(
        f"E153评分复现 V7={v7_summary['score']:.6f} V12={v12_summary['score']:.6f}",
        flush=True,
    )

    v7_angle, v7_delay, v7_log_power = channel_group_maps(
        v7_channel, device, args.batch_size
    )
    v12_angle, v12_delay, v12_log_power = channel_group_maps(
        v12_channel, device, args.batch_size
    )
    truth_angle, truth_delay, truth_log_power = channel_group_maps(
        truth_all, device, args.batch_size, query_ids
    )

    angle_action = 0.5 * np.abs(v12_angle - v7_angle).sum(axis=(1, 2, 3)) / e138.N_UE
    delay_action = (
        0.5
        * np.abs(v12_delay - v7_delay).sum(axis=(1, 2, 3))
        / (e138.MP * e138.N_UE)
    )
    angle_alignment = direction_cosine(
        np.sqrt(v12_angle) - np.sqrt(v7_angle),
        np.sqrt(truth_angle) - np.sqrt(v7_angle),
    )
    delay_alignment = direction_cosine(
        np.sqrt(v12_delay) - np.sqrt(v7_delay),
        np.sqrt(truth_delay) - np.sqrt(v7_delay),
    )
    pas_gain = v12_values[0] - v7_values[0]
    pdp_gain = v12_values[1] - v7_values[1]
    point_nmse_v7 = np.divide(
        v7_values[2], np.maximum(v7_values[3], TINY), out=np.zeros(500), where=v7_values[3] > 0
    )
    point_nmse_v12 = np.divide(
        v12_values[2], np.maximum(v12_values[3], TINY), out=np.zeros(500), where=v12_values[3] > 0
    )
    point_score_gain = (
        0.4 * pas_gain
        + 0.4 * pdp_gain
        + 0.2 * (1.0 / (1.0 + point_nmse_v12) - 1.0 / (1.0 + point_nmse_v7))
    )

    events: list[dict[str, object]] = []
    for row in range(len(query_ids)):
        for ue in range(e138.N_UE):
            events.extend(
                match_angle_events(
                    v7_angle[row, ue],
                    v12_angle[row, ue],
                    truth_angle[row, ue],
                    row,
                    int(query_ids[row]),
                    int(site[row]),
                    ue,
                )
            )
        for polarization in range(e138.MP):
            for ue in range(e138.N_UE):
                events.extend(
                    match_delay_events(
                        v7_delay[row, polarization, ue],
                        v12_delay[row, polarization, ue],
                        truth_delay[row, polarization, ue],
                        row,
                        int(query_ids[row]),
                        int(site[row]),
                        polarization,
                        ue,
                    )
                )
        if (row + 1) % 100 == 0:
            print(f"E153峰匹配 {row + 1}/500", flush=True)
    event_summary, event_point = summarize_events(events, len(query_ids))
    event_details = event_action_details(events)

    carrier_features, carrier_consensus = carrier_uncertainty(
        device, args.batch_size
    )
    train_angle, train_delay, _ = channel_group_maps(
        truth_all, device, args.batch_size
    )
    neighbor_value, neighbor_angle, neighbor_delay = neighbor_features(
        query_ids,
        train_positions,
        train_angle,
        train_delay,
        v7_angle,
        v7_delay,
        v12_angle,
        v12_delay,
    )
    anchor_angle, anchor_delay = compact_anchor_maps()
    anchor_angle_alignment = direction_cosine(
        np.sqrt(anchor_angle) - np.sqrt(v7_angle),
        np.sqrt(v12_angle) - np.sqrt(v7_angle),
    )
    anchor_delay_alignment = direction_cosine(
        np.sqrt(anchor_delay) - np.sqrt(v7_delay),
        np.sqrt(v12_delay) - np.sqrt(v7_delay),
    )

    v7_raw = point_spectral_features(v7_angle, v7_delay, v7_log_power)
    v7_named = {
        "V7角度熵": v7_raw["v7_angle_entropy"],
        "V7角度Top1质量": v7_raw["v7_angle_top1"],
        "V7角度Top4质量": v7_raw["v7_angle_top4"],
        "V7时延熵": v7_raw["v7_delay_entropy"],
        "V7时延Top1质量": v7_raw["v7_delay_top1"],
        "V7时延Top4质量": v7_raw["v7_delay_top4"],
        "V7总功率对数": v7_raw["v7_log_power"],
        "17载体角度分歧": carrier_features["carrier_angle_dispersion"],
        "17载体时延分歧": carrier_features["carrier_delay_dispersion"],
        "17载体功率分歧": carrier_features["carrier_log_power_std"],
        "V7与载体共识角度距离": hellinger_squared(
            v7_angle, carrier_consensus["angle"], (2, 3)
        ).mean(axis=1),
        "V7与载体共识时延距离": hellinger_squared(
            v7_delay, carrier_consensus["delay"], (3,)
        ).mean(axis=(1, 2)),
    }
    neighbor_named = {
        "最近邻距离": neighbor_value["nearest_distance"],
        "前8邻居平均距离": neighbor_value["mean_distance8"],
        "前16邻居平均距离": neighbor_value["mean_distance16"],
        "有效邻居数量": neighbor_value["effective_neighbor_count"],
        "邻居角度分歧": neighbor_value["neighbor_angle_dispersion"],
        "邻居时延分歧": neighbor_value["neighbor_delay_dispersion"],
        "邻居角度修正方向一致性": neighbor_value["neighbor_angle_alignment"],
        "邻居时延修正方向一致性": neighbor_value["neighbor_delay_alignment"],
        "邻居共识与V7角度距离": neighbor_value["neighbor_angle_distance_to_v7"],
        "邻居共识与V7时延距离": neighbor_value["neighbor_delay_distance_to_v7"],
        "官方邻居锚点角度方向一致性": anchor_angle_alignment,
        "官方邻居锚点时延方向一致性": anchor_delay_alignment,
    }
    position_named = {
        "X坐标": geometry_query[:, 0],
        "Y坐标": geometry_query[:, 1],
        "基站距离": geometry_query[:, 6],
        "方位角正弦": geometry_query[:, 8],
        "方位角余弦": geometry_query[:, 9],
        "基站0": geometry_query[:, 12],
        "基站1": geometry_query[:, 13],
    }
    map_named = {
        MAP_LABELS.get(name, name): geometry_query[:, index]
        for index, name in enumerate(geometry_names)
        if index >= 14
    }
    targets = {
        "angle_action": angle_action,
        "delay_action": delay_action,
        "pas_gain": pas_gain,
        "pdp_gain": pdp_gain,
        "angle_new_mass": event_point["angle_new_mass"],
        "delay_new_mass": event_point["delay_new_mass"],
    }
    feature_groups = {
        "V7自身": v7_named,
        "邻居信道": neighbor_named,
        "位置": position_named,
        "地图结构": map_named,
    }
    correlations = feature_correlations(feature_groups, targets)

    matrices = {
        "V7自身": np.column_stack(list(v7_named.values())),
        "邻居信道": np.column_stack(list(neighbor_named.values())),
        "位置": np.column_stack(list(position_named.values())),
        "地图结构": np.column_stack(list(map_named.values())),
    }
    matrices["全部信息"] = np.column_stack(list(matrices.values()))
    predict_targets = {
        key: targets[key]
        for key in ("angle_action", "delay_action", "pas_gain", "pdp_gain")
    }
    predictability = cross_region_predictability(
        matrices, predict_targets, fold_id, args.trees, args.seed
    )
    point_strata = point_strata_summary(
        site, regime, angle_action, delay_action, pas_gain, pdp_gain
    )
    information_quartiles = {
        "v7_angle_top1": quartile_relation(
            v7_named["V7角度Top1质量"], angle_action, pas_gain
        ),
        "v7_delay_top1": quartile_relation(
            v7_named["V7时延Top1质量"], delay_action, pdp_gain
        ),
        "neighbor_angle_dispersion": quartile_relation(
            neighbor_named["邻居角度分歧"], angle_action, pas_gain
        ),
        "neighbor_delay_dispersion": quartile_relation(
            neighbor_named["邻居时延分歧"], delay_action, pdp_gain
        ),
        "map_obstruction_angle_new": quartile_relation(
            map_named["扇形射线平均遮挡数"],
            event_point["angle_new_mass"],
            pas_gain,
        ),
        "map_obstruction_delay_new": quartile_relation(
            map_named["扇形射线平均遮挡数"],
            event_point["delay_new_mass"],
            pdp_gain,
        ),
    }

    point_rows: list[dict[str, object]] = []
    for row in range(len(query_ids)):
        value: dict[str, object] = {
            "point_row": row,
            "query_id": int(query_ids[row]),
            "fold": int(fold_id[row]),
            "site": int(site[row]),
            "regime": REGIME_NAMES[int(regime[row])],
            "x": float(train_positions[query_ids[row], 0]),
            "y": float(train_positions[query_ids[row], 1]),
            "truth_zero": bool(v7_values[3][row] <= TINY),
            "v7_pas": float(v7_values[0][row]),
            "v12_pas": float(v12_values[0][row]),
            "pas_gain": float(pas_gain[row]),
            "v7_pdp": float(v7_values[1][row]),
            "v12_pdp": float(v12_values[1][row]),
            "pdp_gain": float(pdp_gain[row]),
            "point_nmse_v7": float(point_nmse_v7[row]),
            "point_nmse_v12": float(point_nmse_v12[row]),
            "point_score_gain_proxy": float(point_score_gain[row]),
            "angle_action": float(angle_action[row]),
            "delay_action": float(delay_action[row]),
            "angle_direction_alignment": float(angle_alignment[row]),
            "delay_direction_alignment": float(delay_alignment[row]),
        }
        for name, array in event_point.items():
            value[name] = float(array[row])
        for group in feature_groups.values():
            for name, array in group.items():
                value[name] = float(array[row])
        point_rows.append(value)

    write_csv(POINT_CSV, point_rows)
    write_csv(EVENT_CSV, events)
    write_csv(CORRELATION_CSV, correlations)
    write_csv(PREDICTABILITY_CSV, predictability)

    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    figures = [
        score_contribution_figure(v7_summary, v12_summary),
        spatial_gain_figure(
            train_positions, query_ids, site, setup.bs_positions, pas_gain, pdp_gain
        ),
        event_summary_figure(event_summary, events),
        concentration_figure(
            v7_angle, v12_angle, truth_angle, v7_delay, v12_delay, truth_delay
        ),
        binned_relation_figure(feature_groups, targets),
        neighbor_alignment_figure(neighbor_value, pas_gain, pdp_gain),
        map_correlation_figure(correlations),
        predictability_figure(predictability),
        regime_figure(regime, angle_action, delay_action, pas_gain, pdp_gain),
        angle_examples_figure(query_ids, v7_angle, v12_angle, truth_angle, pas_gain),
        delay_examples_figure(query_ids, v7_delay, v12_delay, truth_delay, pdp_gain),
        angle_action_location_figure(events),
        delay_action_location_figure(events),
        information_strata_figure(
            v7_named,
            neighbor_named,
            map_named,
            angle_action,
            delay_action,
            pas_gain,
            pdp_gain,
            event_point,
        ),
    ]

    pas_contribution = 0.4 * (float(v12_summary["pas"]) - float(v7_summary["pas"]))
    pdp_contribution = 0.4 * (float(v12_summary["pdp"]) - float(v7_summary["pdp"]))
    nmse_contribution = 0.2 * (
        float(v12_summary["nmse_score"]) - float(v7_summary["nmse_score"])
    )
    total_contribution = pas_contribution + pdp_contribution + nmse_contribution

    top_correlation = sorted(
        correlations, key=lambda row: abs(float(row["spearman"])), reverse=True
    )[:20]
    top_predictability = sorted(
        predictability, key=lambda row: float(row["spearman"]), reverse=True
    )[:12]
    map_best = max(
        (row for row in predictability if row["group"] == "地图结构"),
        key=lambda row: float(row["spearman"]),
    )
    neighbor_best = max(
        (row for row in predictability if row["group"] == "邻居信道"),
        key=lambda row: float(row["spearman"]),
    )
    v7_best = max(
        (row for row in predictability if row["group"] == "V7自身"),
        key=lambda row: float(row["spearman"]),
    )
    angle_position = {
        str(row["action"]): row for row in event_details["angle"]["position_actions"]
    }
    delay_position = {
        str(row["action"]): row for row in event_details["delay"]["position_actions"]
    }
    angle_site_action = {
        (int(row["site"]), str(row["action"])): row
        for row in event_details["angle"]["position_actions_by_site"]
    }
    delay_site_action = {
        (int(row["site"]), str(row["action"])): row
        for row in event_details["delay"]["position_actions_by_site"]
    }
    low_angle_confidence = information_quartiles["v7_angle_top1"][0]
    high_angle_confidence = information_quartiles["v7_angle_top1"][-1]
    low_neighbor_angle = information_quartiles["neighbor_angle_dispersion"][0]
    high_neighbor_angle = information_quartiles["neighbor_angle_dispersion"][-1]
    low_obstruction = information_quartiles["map_obstruction_angle_new"][0]
    high_obstruction = information_quartiles["map_obstruction_angle_new"][-1]
    heavy = next(row for row in point_strata["regime"] if row["regime"] == "heavy")
    clear = next(row for row in point_strata["regime"] if row["regime"] == "clear")
    neighbor_angle_gain_rho = float(
        spearmanr(neighbor_named["邻居角度修正方向一致性"], pas_gain).statistic
    )
    neighbor_delay_gain_rho = float(
        spearmanr(neighbor_named["邻居时延修正方向一致性"], pdp_gain).statistic
    )

    findings = {
        "pas_score_share": pas_contribution / total_contribution,
        "pdp_score_share": pdp_contribution / total_contribution,
        "nmse_score_share": nmse_contribution / total_contribution,
        "angle_top1_v7_median": float(
            np.median(top_mass(v7_angle, (2, 3), 1).mean(axis=1))
        ),
        "angle_top1_v12_median": float(
            np.median(top_mass(v12_angle, (2, 3), 1).mean(axis=1))
        ),
        "angle_top1_truth_median": float(
            np.median(top_mass(truth_angle, (2, 3), 1).mean(axis=1))
        ),
        "delay_top1_v7_median": float(
            np.median(top_mass(v7_delay, (3,), 1).mean(axis=(1, 2)))
        ),
        "delay_top1_v12_median": float(
            np.median(top_mass(v12_delay, (3,), 1).mean(axis=(1, 2)))
        ),
        "delay_top1_truth_median": float(
            np.median(top_mass(truth_delay, (3,), 1).mean(axis=(1, 2)))
        ),
        "angle_direction_alignment_median": float(np.median(angle_alignment)),
        "delay_direction_alignment_median": float(np.median(delay_alignment)),
        "primary_conclusion": (
            "V12的主要收益来自角域重分配：把V7分散或放错位置的能量向少数主峰集中。"
            f"角度移动和消失的局部改善率分别为{float(angle_position['移动']['beneficial_fraction']):.1%}/"
            f"{float(angle_position['消失']['beneficial_fraction']):.1%}，但新增只有"
            f"{float(angle_position['新增']['beneficial_fraction']):.1%}；"
            "因此有效动作更接近移动、压低和删除假峰，而不是普遍补峰。"
        ),
        "uncertainty_conclusion": (
            f"V7自身信息对标量修正目标的最佳跨区域Spearman为"
            f"{float(v7_best['spearman']):+.3f}（{TARGET_LABELS[str(v7_best['target'])]}），"
            f"V7角度Top1最低四分位的平均修正/ PAS增益为"
            f"{float(low_angle_confidence['target_mean']):.3f}/{float(low_angle_confidence['gain_mean']):+.3f}，"
            f"最高四分位为{float(high_angle_confidence['target_mean']):.3f}/"
            f"{float(high_angle_confidence['gain_mean']):+.3f}。"
            "因此V7置信度能较好回答哪里需要修，但仍不能给出峰应移动到哪里。"
        ),
        "neighbor_conclusion": (
            f"邻居信道信息的最佳跨区域Spearman为"
            f"{float(neighbor_best['spearman']):+.3f}（{TARGET_LABELS[str(neighbor_best['target'])]}），"
            f"邻居角度分歧最低到最高四分位时修正幅度由"
            f"{float(low_neighbor_angle['target_mean']):.3f}升至"
            f"{float(high_neighbor_angle['target_mean']):.3f}；"
            f"但邻居方向一致性与PAS/PDP增益的相关仅为"
            f"{neighbor_angle_gain_rho:+.3f}/{neighbor_delay_gain_rho:+.3f}。"
            "邻居更像不稳定性报警器，不能直接充当修正方向。"
        ),
        "map_conclusion": (
            f"地图结构信息的最佳跨区域Spearman为"
            f"{float(map_best['spearman']):+.3f}（{TARGET_LABELS[str(map_best['target'])]}）。"
            f"遮挡最低到最高四分位时，角度新增质量由"
            f"{float(low_obstruction['target_mean']):.3f}升至"
            f"{float(high_obstruction['target_mean']):.3f}，PAS增益却由"
            f"{float(low_obstruction['gain_mean']):+.3f}降至"
            f"{float(high_obstruction['gain_mean']):+.3f}；"
            f"heavy区角度动作比clear区更大（{float(heavy['angle_action_mean']):.3f}对"
            f"{float(clear['angle_action_mean']):.3f}），但PAS/PDP增益只有"
            f"{float(heavy['pas_gain_mean']):+.3f}/{float(heavy['pdp_gain_mean']):+.3f}。"
            "地图能说明复杂度和补峰风险，但不能单独决定精确峰状态。"
        ),
        "site_asymmetry_conclusion": (
            f"新增错误高度集中在基站1：角度新增{int(angle_site_action[(1, '新增')]['count'])}次，"
            f"改善率{float(angle_site_action[(1, '新增')]['beneficial_fraction']):.1%}；"
            f"时延新增{int(delay_site_action[(1, '新增')]['count'])}次，"
            f"改善率{float(delay_site_action[(1, '新增')]['beneficial_fraction']):.1%}。"
            f"基站0对应为{int(angle_site_action[(0, '新增')]['count'])}/"
            f"{float(angle_site_action[(0, '新增')]['beneficial_fraction']):.1%}和"
            f"{int(delay_site_action[(0, '新增')]['count'])}/"
            f"{float(delay_site_action[(0, '新增')]['beneficial_fraction']):.1%}。"
        ),
        "delay_conclusion": (
            f"PDP中{float(delay_position['保留']['fraction']):.1%}的峰位置保留，"
            f"移动峰有{int(delay_position['移动']['count'])}个，其中"
            f"{int(event_details['delay']['move_distribution'][0]['count'])}个只移动1格；"
            f"新增峰局部改善率仅{float(delay_position['新增']['beneficial_fraction']):.1%}。"
            "因此PDP收益主要是已有早期主峰的一格对齐和幅度重标定，不是恢复大量新晚到路径。"
        ),
    }

    report = {
        "experiment": VERSION,
        "status": "analysis_complete_no_model",
        "goal": (
            "Identify V7-to-V12 angle/delay peak actions and quantify their relationship "
            "with V7 uncertainty, official neighbors, and map structure."
        ),
        "protocol": {
            "canonical_500": True,
            "official_metric": True,
            "solution_model_trained": False,
            "submission_generated": False,
            "angle_peak_definition": {
                "maps": "frequency-averaged official PAS, polarization-collapsed per UE",
                "minimum_peak_mass": ANGLE_PEAK_MIN,
                "local_window": "periodic 3x3",
                "match_tolerance": "2 horizontal bins and 1 vertical bin",
            },
            "delay_peak_definition": {
                "maps": "official PDP averaged over H/V per polarization and UE",
                "minimum_peak_value": DELAY_PEAK_MIN,
                "local_window": "periodic +/-2 delay bins",
                "match_tolerance_bins": DELAY_MATCH_BINS,
            },
            "v7_uncertainty": "17 official-input OOF complex carriers",
            "neighbor_information": "64 strict official OOF neighbors",
            "map_information": "official geometry visibility features",
        },
        "score_reproduction": {
            "v7": v7_summary,
            "v12": v12_summary,
            "delta": float(v12_summary["score"] - v7_summary["score"]),
            "weighted_contribution": {
                "pas": pas_contribution,
                "pdp": pdp_contribution,
                "nmse_score": nmse_contribution,
            },
        },
        "peak_events": event_summary,
        "peak_event_details": event_details,
        "point_strata": point_strata,
        "information_quartiles": information_quartiles,
        "point_action_summary": {
            "angle_action_mean_median_p90": [
                float(np.mean(angle_action)),
                float(np.median(angle_action)),
                float(np.quantile(angle_action, 0.90)),
            ],
            "delay_action_mean_median_p90": [
                float(np.mean(delay_action)),
                float(np.median(delay_action)),
                float(np.quantile(delay_action, 0.90)),
            ],
            "pas_positive_points": int(np.sum(pas_gain > 0.0)),
            "pdp_positive_points": int(np.sum(pdp_gain > 0.0)),
            "angle_direction_positive_points": int(np.sum(angle_alignment > 0.0)),
            "delay_direction_positive_points": int(np.sum(delay_alignment > 0.0)),
        },
        "top_correlations": top_correlation,
        "cross_region_predictability": predictability,
        "top_predictability": top_predictability,
        "findings": findings,
        "files": {
            "point_csv": PATHS.relative(POINT_CSV),
            "event_csv": PATHS.relative(EVENT_CSV),
            "correlation_csv": PATHS.relative(CORRELATION_CSV),
            "predictability_csv": PATHS.relative(PREDICTABILITY_CSV),
            "metrics": PATHS.relative(METRICS),
            "markdown": PATHS.relative(REPORT_MD),
            "figures": [PATHS.relative(path) for path in figures],
        },
        "font": font_name,
        "seconds": float(time.time() - started),
    }

    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        site=site,
        regime=regime,
        v7_pas=v7_values[0],
        v12_pas=v12_values[0],
        v7_pdp=v7_values[1],
        v12_pdp=v12_values[1],
        v7_error=v7_values[2],
        v12_error=v12_values[2],
        truth_energy=v7_values[3],
        pas_gain=pas_gain,
        pdp_gain=pdp_gain,
        point_score_gain_proxy=point_score_gain,
        angle_action=angle_action,
        delay_action=delay_action,
        angle_alignment=angle_alignment,
        delay_alignment=delay_alignment,
        v7_angle=v7_angle.astype(np.float16),
        v12_angle=v12_angle.astype(np.float16),
        truth_angle=truth_angle.astype(np.float16),
        v7_delay=v7_delay.astype(np.float16),
        v12_delay=v12_delay.astype(np.float16),
        truth_delay=truth_delay.astype(np.float16),
        neighbor_angle=neighbor_angle.astype(np.float16),
        neighbor_delay=neighbor_delay.astype(np.float16),
        **{name: value.astype(np.float32) for name, value in event_point.items()},
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report, figures), encoding="utf-8")
    print(
        json.dumps(
            {
                "v7": v7_summary,
                "v12": v12_summary,
                "event_summary": event_summary,
                "findings": findings,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
