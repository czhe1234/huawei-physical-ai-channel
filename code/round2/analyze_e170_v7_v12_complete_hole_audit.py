#!/usr/bin/env python3
"""Audit canonical V7-V12 scores, hole strata, and PAS peak evolution."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
import torch

import analyze_e167_clean_test_hole_difficulty as e167
import analyze_e169_hard_hole_pas_peak_precursors as e169
from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics,
)
from project_paths import PATHS
from run_v8_full_sionna_migration import Action, apply_action


VERSION = "Round2-E170-v7-v12-complete-hole-audit"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e170_v7_v12_complete_hole_audit"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e170_v7_v12_complete_hole_audit"
VIS_DIR = PATHS.analysis / "visualizations" / "复赛V7到V12完整洞审计"

REPORT_JSON = ANALYSIS_DIR / "v7_v12_complete_hole_audit.json"
REPORT_MD = ANALYSIS_DIR / "V7到V12_fixed500完整审计.md"
OVERALL_CSV = ANALYSIS_DIR / "V7到V12整体官方分.csv"
CONTRIBUTION_CSV = ANALYSIS_DIR / "逐版本官方分贡献分解.csv"
HOLE_CSV = ANALYSIS_DIR / "十个pseudo洞逐版本官方分.csv"
DIFFICULTY_CSV = ANALYSIS_DIR / "难度分层逐版本官方分.csv"
DEPTH_CSV = ANALYSIS_DIR / "洞深度分层逐版本官方分.csv"
ERROR_CSV = ANALYSIS_DIR / "PAS峰错误逐版本分层.csv"
ACTION_CSV = ANALYSIS_DIR / "PAS峰动作逐版本分层.csv"
CORRECTION_CSV = ANALYSIS_DIR / "PAS修正幅度与方向.csv"
REPAIR_CSV = ANALYSIS_DIR / "V12对V7峰错误修复率.csv"
POINT_CSV = ANALYSIS_DIR / "V12相对V7逐点修复与破坏.csv"
POINT_SUMMARY_CSV = ANALYSIS_DIR / "V12逐点修复破坏分层汇总.csv"

TRAIN_CHANNEL = e167.TRAIN_CHANNEL
OOF_META = e167.OOF_META
DEPTH_SOURCE = e167.POINT_CSV

V8_CHANNEL = ARTIFACT_DIR / "V8_Canonical_500_Reconstructed.npy"
V8_BASE = PATHS.artifacts / "round2" / "full_sionna_migration" / "oof_v7_nested_channel.npy"
V8_PAS = PATHS.artifacts / "round2" / "sionna_official_calibration" / "oof_pas_mean_f16.npy"
V8_PDP = PATHS.artifacts / "round2" / "sionna_official_calibration" / "oof_pdp_mean_f16.npy"

CHANNEL_PATHS = {
    "V7": e167.V7_COMPLETE,
    "V8": V8_CHANNEL,
    "V9": PATHS.artifacts / "round2" / "canonical_500_version_replay" / "V9_Canonical_500_OOF_Channel.npy",
    "V10": PATHS.artifacts / "round2" / "canonical_500_version_replay" / "V10_Canonical_500_OOF_Channel.npy",
    "V11": PATHS.artifacts / "round2" / "canonical_500_version_replay" / "V11_Canonical_500_OOF_Channel.npy",
    "V12": PATHS.artifacts / "round2" / "surface_interaction_energy_model" / "e86_frozen_geometry_sequence_oof_channel.npy",
}

EXPECTED = {
    "V7": {"pas": 0.597983259844433, "pdp": 0.7564459407811781, "nmse": 0.9477104887777993, "score": 0.6444563457343335},
    "V8": {"pas": 0.6253015427342626, "pdp": 0.7634162327000106, "nmse": 0.9405859252300354, "score": 0.6585487666557804},
    "V9": {"pas": 0.644791115088469, "pdp": 0.7650376243394221, "nmse": 0.9378234316029705, "score": 0.667140073363058},
    "V10": {"pas": 0.6455208038684359, "pdp": 0.7678360427453826, "nmse": 0.9371669299543997, "score": 0.6685862933993286},
    "V11": {"pas": 0.6480477941511763, "pdp": 0.7679746548310322, "nmse": 0.9360367600034644, "score": 0.6697128032246425},
    "V12": {"pas": 0.6625825788, "pdp": 0.7727009798, "nmse": 0.9340978586, "score": 0.6775208075},
}

ONLINE = {
    "V7": 0.636929,
    "V8": 0.662169,
    "V9": 0.675445,
    "V10": 0.679708,
    "V11": 0.682145,
    "V12": 0.685857,
}

VERSIONS = tuple(EXPECTED)
IDENTITY_TOLERANCE = 5e-6
TINY = 1e-30
EASY_HOLES = ("P_S0F4", "P_S0F0", "P_S1F0")
MEDIUM_HOLES = ("P_S0F1", "P_S0F3", "P_S0F2")
HARD_HOLES = ("P_S1F3", "P_S1F1", "P_S1F2", "P_S1F4")
DIFFICULTY_HOLES = {
    "easy": EASY_HOLES,
    "medium": MEDIUM_HOLES,
    "hard": HARD_HOLES,
}
DEPTH_ORDER = ("边缘", "中部", "深处")
ERROR_TYPES = ("missing_peak", "spurious_peak", "angle_shift", "power_mismatch")
ERROR_NAMES = {
    "missing_peak": "漏峰",
    "spurious_peak": "假峰",
    "angle_shift": "角度偏移",
    "power_mismatch": "功率不匹配",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fft-batch-size", type=int, default=16)
    parser.add_argument("--force-v8", action="store_true")
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
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def validate_channel(path: Path) -> np.memmap:
    channel = np.load(path, mmap_mode="r")
    if channel.shape != (500, 256, 4, 192) or channel.dtype != np.complex64:
        raise ValueError(f"invalid channel {PATHS.relative(path)}: {channel.shape}/{channel.dtype}")
    for start in range(0, len(channel), 16):
        block = np.asarray(channel[start : start + 16])
        if not np.isfinite(block.real).all() or not np.isfinite(block.imag).all():
            raise ValueError(f"non-finite channel values in {PATHS.relative(path)}")
    return channel


def reconstruct_v8(device: torch.device, batch_size: int, force: bool) -> np.memmap:
    if V8_CHANNEL.exists() and not force:
        return validate_channel(V8_CHANNEL)
    base = validate_channel(V8_BASE)
    pas = np.load(V8_PAS, mmap_mode="r")
    pdp = np.load(V8_PDP, mmap_mode="r")
    if pas.shape != (500, 2, 16, 8, 4, 192):
        raise ValueError(f"unexpected V8 PAS endpoint shape: {pas.shape}")
    if pdp.shape != (500, 256, 4, 192):
        raise ValueError(f"unexpected V8 PDP endpoint shape: {pdp.shape}")
    output = np.lib.format.open_memmap(
        V8_CHANNEL, mode="w+", dtype=np.complex64, shape=(500, 256, 4, 192)
    )
    action = Action(
        "joint_a030_030",
        alpha_pas=0.30,
        alpha_pdp=0.30,
        order="pas_pdp",
        cycles=4,
    )
    for start in range(0, 500, batch_size):
        stop = min(start + batch_size, 500)
        base_tensor = torch.as_tensor(
            np.array(base[start:stop], dtype=np.complex64, copy=True), device=device
        )
        pas_tensor = torch.as_tensor(
            np.array(pas[start:stop], dtype=np.float32, copy=True), device=device
        )
        pdp_tensor = torch.as_tensor(
            np.array(pdp[start:stop], dtype=np.float32, copy=True), device=device
        )
        with torch.inference_mode():
            candidate = apply_action(
                base_tensor,
                pas_tensor,
                pdp_tensor,
                torch.empty(0, device=device),
                np.zeros(stop - start, dtype=np.int8),
                action,
            )
        output[start:stop] = candidate.cpu().numpy().astype(np.complex64, copy=False)
        if stop % 100 == 0 or stop == 500:
            output.flush()
            print(f"E170 reconstruct V8: {stop}/500", flush=True)
    output.flush()
    return validate_channel(V8_CHANNEL)


def score_channel(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    batch_size: int,
) -> tuple[PointMetricStatistics, dict[str, float | int]]:
    pas = np.empty(len(query_ids), dtype=np.float64)
    pdp = np.empty(len(query_ids), dtype=np.float64)
    squared_error = np.empty(len(query_ids), dtype=np.float64)
    truth_energy = np.empty(len(query_ids), dtype=np.float64)
    for start in range(0, len(query_ids), batch_size):
        stop = min(start + batch_size, len(query_ids))
        values = point_metric_statistics(
            np.asarray(prediction[start:stop], dtype=np.complex64),
            np.asarray(truth[query_ids[start:stop]], dtype=np.complex64),
        )
        pas[start:stop] = values.pas
        pdp[start:stop] = values.pdp
        squared_error[start:stop] = values.squared_error
        truth_energy[start:stop] = values.truth_energy
    statistics = PointMetricStatistics(pas, pdp, squared_error, truth_energy)
    return statistics, aggregate_statistics(statistics)


def identity_errors(version: str, overall: dict[str, float | int]) -> dict[str, float]:
    errors = {
        key: abs(float(overall[key]) - float(EXPECTED[version][key]))
        for key in ("pas", "pdp", "nmse", "score")
    }
    if max(errors.values()) > IDENTITY_TOLERANCE:
        raise RuntimeError(f"{version} identity gate failed: {errors}")
    return errors


def load_depth_labels() -> tuple[np.ndarray, np.ndarray]:
    depth = np.empty(500, dtype=object)
    global_id = np.empty(500, dtype=np.int64)
    seen = np.zeros(500, dtype=bool)
    with DEPTH_SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["oof_row"])
            depth[index] = row["depth_bin"]
            global_id[index] = int(row["global_train_id"])
            seen[index] = True
    if not np.all(seen):
        raise RuntimeError("E167 depth labels do not cover all 500 OOF points")
    return depth, global_id


def build_point_labels(
    query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    _, holes, context = e167.build_holes()
    train_positions = np.asarray(context["train_positions"], dtype=np.float64)
    global_to_oof = np.full(len(train_positions), -1, dtype=np.int64)
    global_to_oof[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    hole = np.empty(500, dtype=object)
    site = np.empty(500, dtype=np.int8)
    fold = np.empty(500, dtype=np.int8)
    hole_masks: dict[str, np.ndarray] = {}
    for item in holes:
        rows = global_to_oof[item.point_ids]
        if np.any(rows < 0):
            raise RuntimeError(f"hole {item.hole_id} escaped canonical OOF")
        hole[rows] = item.hole_id
        site[rows] = item.site
        fold[rows] = item.ordinal
        mask = np.zeros(500, dtype=bool)
        mask[rows] = True
        hole_masks[item.hole_id] = mask
    difficulty = np.empty(500, dtype=object)
    for label, members in DIFFICULTY_HOLES.items():
        difficulty[np.isin(hole, members)] = label
    if any(value not in DIFFICULTY_HOLES for value in difficulty):
        raise RuntimeError("difficulty labels are incomplete")
    return hole, difficulty, site, fold, hole_masks


def aggregate_mask(
    statistics: PointMetricStatistics,
    mask: np.ndarray,
    version: str,
    **labels: object,
) -> dict[str, object]:
    return {"version": version, **labels, **aggregate_statistics(statistics, mask)}


def score_tables(
    statistics: dict[str, PointMetricStatistics],
    overalls: dict[str, dict[str, float | int]],
    hole: np.ndarray,
    difficulty: np.ndarray,
    depth: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    overall_rows: list[dict[str, object]] = []
    hole_rows: list[dict[str, object]] = []
    difficulty_rows: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []
    for version in VERSIONS:
        overall = overalls[version]
        overall_rows.append(
            {
                "version": version,
                **overall,
                "delta_pas_vs_v7": float(overall["pas"]) - float(overalls["V7"]["pas"]),
                "delta_pdp_vs_v7": float(overall["pdp"]) - float(overalls["V7"]["pdp"]),
                "delta_nmse_vs_v7": float(overall["nmse"]) - float(overalls["V7"]["nmse"]),
                "delta_score_vs_v7": float(overall["score"]) - float(overalls["V7"]["score"]),
                "online_score": ONLINE[version],
            }
        )
        for hole_id in sorted(np.unique(hole).tolist()):
            row = aggregate_mask(
                statistics[version], hole == hole_id, version, pseudo_hole_id=hole_id
            )
            row["difficulty"] = str(difficulty[np.flatnonzero(hole == hole_id)[0]])
            hole_rows.append(row)
        for label in ("easy", "medium", "hard"):
            difficulty_rows.append(
                aggregate_mask(
                    statistics[version], difficulty == label, version, difficulty=label
                )
            )
        for label in DEPTH_ORDER:
            depth_rows.append(
                aggregate_mask(statistics[version], depth == label, version, depth_bin=label)
            )
    v7_holes = {row["pseudo_hole_id"]: row for row in hole_rows if row["version"] == "V7"}
    v7_difficulty = {row["difficulty"]: row for row in difficulty_rows if row["version"] == "V7"}
    v7_depth = {row["depth_bin"]: row for row in depth_rows if row["version"] == "V7"}
    for row in hole_rows:
        base = v7_holes[row["pseudo_hole_id"]]
        row["delta_pas_vs_v7"] = float(row["pas"]) - float(base["pas"])
        row["delta_pdp_vs_v7"] = float(row["pdp"]) - float(base["pdp"])
        row["delta_nmse_vs_v7"] = float(row["nmse"]) - float(base["nmse"])
        row["delta_score_vs_v7"] = float(row["score"]) - float(base["score"])
    for row in difficulty_rows:
        base = v7_difficulty[row["difficulty"]]
        row["delta_pas_vs_v7"] = float(row["pas"]) - float(base["pas"])
        row["delta_pdp_vs_v7"] = float(row["pdp"]) - float(base["pdp"])
        row["delta_nmse_vs_v7"] = float(row["nmse"]) - float(base["nmse"])
        row["delta_score_vs_v7"] = float(row["score"]) - float(base["score"])
    for row in depth_rows:
        base = v7_depth[row["depth_bin"]]
        row["delta_pas_vs_v7"] = float(row["pas"]) - float(base["pas"])
        row["delta_pdp_vs_v7"] = float(row["pdp"]) - float(base["pdp"])
        row["delta_nmse_vs_v7"] = float(row["nmse"]) - float(base["nmse"])
        row["delta_score_vs_v7"] = float(row["score"]) - float(base["score"])

    for rows, key in (
        (hole_rows, "pseudo_hole_id"),
        (difficulty_rows, "difficulty"),
        (depth_rows, "depth_bin"),
    ):
        indexed = {(str(row["version"]), str(row[key])): row for row in rows}
        for row in rows:
            version = str(row["version"])
            if version == "V7":
                parent = "V7"
            else:
                parent = VERSIONS[VERSIONS.index(version) - 1]
            base = indexed[(parent, str(row[key]))]
            row["parent"] = parent
            row["delta_pas_vs_parent"] = float(row["pas"]) - float(base["pas"])
            row["delta_pdp_vs_parent"] = float(row["pdp"]) - float(base["pdp"])
            row["delta_nmse_vs_parent"] = float(row["nmse"]) - float(base["nmse"])
            row["delta_score_vs_parent"] = float(row["score"]) - float(base["score"])
    return overall_rows, hole_rows, difficulty_rows, depth_rows


def contribution_rows(overalls: dict[str, dict[str, float | int]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons = [("V7", version, "vs_v7") for version in VERSIONS[1:]]
    comparisons += list(zip(VERSIONS[:-1], VERSIONS[1:], ["step"] * (len(VERSIONS) - 1)))
    seen: set[tuple[str, str, str]] = set()
    for base, changed, kind in comparisons:
        key = (base, changed, kind)
        if key in seen:
            continue
        seen.add(key)
        left = overalls[base]
        right = overalls[changed]
        pas = 0.4 * (float(right["pas"]) - float(left["pas"]))
        pdp = 0.4 * (float(right["pdp"]) - float(left["pdp"]))
        nmse = 0.2 * (float(right["nmse_score"]) - float(left["nmse_score"]))
        total = pas + pdp + nmse
        rows.append(
            {
                "comparison_type": kind,
                "base": base,
                "changed": changed,
                "delta_pas": float(right["pas"]) - float(left["pas"]),
                "delta_pdp": float(right["pdp"]) - float(left["pdp"]),
                "delta_nmse": float(right["nmse"]) - float(left["nmse"]),
                "pas_score_contribution": pas,
                "pdp_score_contribution": pdp,
                "nmse_score_contribution": nmse,
                "delta_score": total,
                "pas_share_of_delta": pas / total if total != 0.0 else float("nan"),
                "pdp_share_of_delta": pdp / total if total != 0.0 else float("nan"),
                "nmse_share_of_delta": nmse / total if total != 0.0 else float("nan"),
            }
        )
    return rows


def compact_path(version: str) -> Path:
    if version == "V7":
        return e169.V7_COMPACT
    return ARTIFACT_DIR / f"{version}_compact_pas.npy"


def build_compact_maps(
    channels: dict[str, np.ndarray],
    truth: np.ndarray,
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    train_compact = e169.load_or_build_compact(
        truth, e169.TRAIN_COMPACT, device, batch_size, rebuild=False
    )
    truth_compact = np.asarray(train_compact[query_ids], dtype=np.float32)
    compact: dict[str, np.ndarray] = {}
    for version in VERSIONS:
        compact[version] = e169.load_or_build_compact(
            channels[version], compact_path(version), device, batch_size, rebuild
        )
    return compact, truth_compact


def representation_gate(
    compact: dict[str, np.ndarray],
    truth_compact: np.ndarray,
    statistics: dict[str, PointMetricStatistics],
) -> list[dict[str, object]]:
    rows = []
    for version in VERSIONS:
        proxy = e169.compact_point_cosine(compact[version], truth_compact)
        pearson = float(pearsonr(proxy, statistics[version].pas).statistic)
        spearman = float(spearmanr(proxy, statistics[version].pas).statistic)
        passed = pearson >= 0.70
        rows.append(
            {"version": version, "pearson": pearson, "spearman": spearman, "passed": passed}
        )
        if not passed:
            raise RuntimeError(f"{version} compact PAS representation gate failed: {pearson}")
    return rows


def build_error_events(
    version: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    hole: np.ndarray,
    difficulty: np.ndarray,
    depth: np.ndarray,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for point in range(500):
        for ue in range(e169.N_UE):
            for event in e169.match_peaks(prediction[point, ue], truth[point, ue]):
                event.update(
                    {
                        "version": version,
                        "oof_row": point,
                        "ue": ue,
                        "pseudo_hole_id": str(hole[point]),
                        "difficulty": str(difficulty[point]),
                        "depth_bin": str(depth[point]),
                    }
                )
                events.append(event)
    return events


def summarize_errors(events: list[dict[str, object]], point_count: int) -> dict[str, object]:
    selected = [event for event in events if event["event_type"] in ERROR_TYPES]
    total = sum(float(event["error_weight"]) for event in selected)
    result: dict[str, object] = {
        "error_event_count": len(selected),
        "error_weight_sum": total,
        "error_weight_per_point_ue": total / max(point_count * e169.N_UE, 1),
    }
    for error_type in ERROR_TYPES:
        subset = [event for event in selected if event["event_type"] == error_type]
        weight = sum(float(event["error_weight"]) for event in subset)
        result[f"{error_type}_count"] = len(subset)
        result[f"{error_type}_weight"] = weight
        result[f"{error_type}_share"] = weight / total if total > 0.0 else 0.0
    return result


def error_summary_rows(
    events_by_version: dict[str, list[dict[str, object]]],
    hole: np.ndarray,
    difficulty: np.ndarray,
    depth: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for version in VERSIONS:
        events = events_by_version[version]
        scopes: list[tuple[str, str, int, list[dict[str, object]]]] = [("overall", "ALL", 500, events)]
        for hole_id in sorted(EASY_HOLES + MEDIUM_HOLES + HARD_HOLES):
            scopes.append(("hole", hole_id, int(np.sum(hole == hole_id)), [e for e in events if e["pseudo_hole_id"] == hole_id]))
        for label in ("easy", "medium", "hard"):
            scopes.append(("difficulty", label, int(np.sum(difficulty == label)), [e for e in events if e["difficulty"] == label]))
        for label in DEPTH_ORDER:
            scopes.append(("depth", label, int(np.sum(depth == label)), [e for e in events if e["depth_bin"] == label]))
        for scope_type, scope, point_count, selected in scopes:
            rows.append(
                {
                    "version": version,
                    "scope_type": scope_type,
                    "scope": scope,
                    **summarize_errors(selected, point_count),
                }
            )
    return rows


def peak_action_events(base: np.ndarray, changed: np.ndarray, truth: np.ndarray) -> list[dict[str, object]]:
    old_peaks = e169.extract_peaks(base)
    new_peaks = e169.extract_peaks(changed)
    accepted: list[tuple[int, int, float, float, float]] = []
    if old_peaks and new_peaks:
        cost = np.empty((len(old_peaks), len(new_peaks)), dtype=np.float64)
        axes: dict[tuple[int, int], tuple[float, float]] = {}
        for old_index, old in enumerate(old_peaks):
            for new_index, new in enumerate(new_peaks):
                value, dh, dv = e169.scaled_distance(old, new)
                cost[old_index, new_index] = value
                axes[(old_index, new_index)] = (dh, dv)
        rows, columns = linear_sum_assignment(cost)
        for old_index, new_index in zip(rows, columns, strict=True):
            if cost[old_index, new_index] <= 1.0:
                dh, dv = axes[(int(old_index), int(new_index))]
                accepted.append((int(old_index), int(new_index), float(cost[old_index, new_index]), dh, dv))
    old_matched = {item[0] for item in accepted}
    new_matched = {item[1] for item in accepted}
    events: list[dict[str, object]] = []

    def local_error(mask: np.ndarray, value: np.ndarray) -> float:
        return float(np.abs(value[mask] - truth[mask]).sum(dtype=np.float64))

    for old_index, new_index, _, dh, dv in accepted:
        old = old_peaks[old_index]
        new = new_peaks[new_index]
        mask = np.zeros((e169.N_H, e169.N_V), dtype=bool)
        for peak in (old, new):
            mask[np.ix_(e169.angle_indices(peak.h, 1, e169.N_H), e169.angle_indices(peak.v, 1, e169.N_V))] = True
        tolerance = max(e169.MASS_ABS_TOL, 0.05 * max(old.mass, new.mass))
        if new.mass - old.mass > tolerance:
            mass_action = "increased"
        elif old.mass - new.mass > tolerance:
            mass_action = "decreased"
        else:
            mass_action = "stable"
        events.append(
            {
                "position_action": "moved" if dh > 0.0 or dv > 0.0 else "retained",
                "mass_action": mass_action,
                "old_mass": old.mass,
                "new_mass": new.mass,
                "position_mass": 0.5 * (old.mass + new.mass),
                "reweighted_mass": abs(new.mass - old.mass),
                "beneficial": local_error(mask, changed) < local_error(mask, base),
            }
        )
    for old_index, old in enumerate(old_peaks):
        if old_index in old_matched:
            continue
        mask = np.zeros((e169.N_H, e169.N_V), dtype=bool)
        mask[np.ix_(e169.angle_indices(old.h, 1, e169.N_H), e169.angle_indices(old.v, 1, e169.N_V))] = True
        events.append(
            {
                "position_action": "deleted",
                "mass_action": "decreased",
                "old_mass": old.mass,
                "new_mass": 0.0,
                "position_mass": old.mass,
                "reweighted_mass": old.mass,
                "beneficial": local_error(mask, changed) < local_error(mask, base),
            }
        )
    for new_index, new in enumerate(new_peaks):
        if new_index in new_matched:
            continue
        mask = np.zeros((e169.N_H, e169.N_V), dtype=bool)
        mask[np.ix_(e169.angle_indices(new.h, 1, e169.N_H), e169.angle_indices(new.v, 1, e169.N_V))] = True
        events.append(
            {
                "position_action": "added",
                "mass_action": "increased",
                "old_mass": 0.0,
                "new_mass": new.mass,
                "position_mass": new.mass,
                "reweighted_mass": new.mass,
                "beneficial": local_error(mask, changed) < local_error(mask, base),
            }
        )
    return events


def summarize_actions(events: list[dict[str, object]], point_count: int) -> dict[str, object]:
    output: dict[str, object] = {"point_count": point_count, "event_count": len(events)}
    for action in ("retained", "moved", "deleted", "added"):
        selected = [event for event in events if event["position_action"] == action]
        output[f"{action}_count"] = len(selected)
        output[f"{action}_mass"] = sum(float(event["position_mass"]) for event in selected)
        output[f"{action}_beneficial_rate"] = (
            float(np.mean([bool(event["beneficial"]) for event in selected])) if selected else float("nan")
        )
    for action in ("increased", "decreased"):
        selected = [event for event in events if event["mass_action"] == action]
        output[f"{action}_count"] = len(selected)
        output[f"{action}_absolute_mass"] = sum(float(event["reweighted_mass"]) for event in selected)
    output["reweighted_absolute_mass"] = sum(float(event["reweighted_mass"]) for event in events)
    output["beneficial_event_rate"] = float(np.mean([bool(event["beneficial"]) for event in events])) if events else float("nan")
    return output


def action_summary_rows(
    compact: dict[str, np.ndarray],
    truth: np.ndarray,
    difficulty: np.ndarray,
) -> list[dict[str, object]]:
    comparisons = [
        ("V7", "V8", "step_and_vs_v7"),
        ("V7", "V9", "vs_v7"),
        ("V7", "V10", "vs_v7"),
        ("V7", "V11", "vs_v7"),
        ("V7", "V12", "vs_v7"),
        ("V8", "V9", "step"),
        ("V9", "V10", "step"),
        ("V10", "V11", "step"),
        ("V11", "V12", "step"),
    ]
    rows: list[dict[str, object]] = []
    for base_version, changed_version, kind in comparisons:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for point in range(500):
            for ue in range(e169.N_UE):
                events = peak_action_events(
                    compact[base_version][point, ue],
                    compact[changed_version][point, ue],
                    truth[point, ue],
                )
                grouped["ALL"].extend(events)
                grouped[str(difficulty[point])].extend(events)
        point_tv = 0.5 * np.abs(
            compact[changed_version] - compact[base_version]
        ).sum(axis=(1, 2, 3), dtype=np.float64) / e169.N_UE
        for scope in ("ALL", "easy", "medium", "hard"):
            mask = np.ones(500, dtype=bool) if scope == "ALL" else difficulty == scope
            rows.append(
                {
                    "comparison_type": kind,
                    "base": base_version,
                    "changed": changed_version,
                    "difficulty": scope,
                    "mean_compact_pas_total_variation": float(np.mean(point_tv[mask])),
                    "p90_compact_pas_total_variation": float(np.quantile(point_tv[mask], 0.90)),
                    **summarize_actions(grouped[scope], int(np.sum(mask))),
                }
            )
    return rows


def direction_cosine(delta: np.ndarray, target: np.ndarray) -> np.ndarray:
    left = delta.reshape(len(delta), -1).astype(np.float64)
    right = target.reshape(len(target), -1).astype(np.float64)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > TINY)


def correction_rows(
    compact: dict[str, np.ndarray],
    truth: np.ndarray,
    statistics: dict[str, PointMetricStatistics],
    hole: np.ndarray,
    difficulty: np.ndarray,
    depth: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = compact["V7"]
    target = truth - base
    for version in VERSIONS:
        delta = compact[version] - base
        tv = 0.5 * np.abs(delta).sum(axis=(1, 2, 3), dtype=np.float64) / e169.N_UE
        cosine = direction_cosine(delta, target)
        pas_delta = statistics[version].pas - statistics["V7"].pas
        scopes: list[tuple[str, str, np.ndarray]] = [("overall", "ALL", np.ones(500, dtype=bool))]
        scopes += [("hole", value, hole == value) for value in sorted(np.unique(hole).tolist())]
        scopes += [("difficulty", value, difficulty == value) for value in ("easy", "medium", "hard")]
        scopes += [("depth", value, depth == value) for value in DEPTH_ORDER]
        for scope_type, scope, mask in scopes:
            changed = tv[mask] > 1e-12
            rows.append(
                {
                    "version": version,
                    "scope_type": scope_type,
                    "scope": scope,
                    "count": int(np.sum(mask)),
                    "changed_count": int(np.sum(changed)),
                    "mean_compact_pas_total_variation": float(np.mean(tv[mask])),
                    "p90_compact_pas_total_variation": float(np.quantile(tv[mask], 0.90)),
                    "mean_direction_cosine_truth_minus_v7": float(np.mean(cosine[mask][changed])) if np.any(changed) else 0.0,
                    "mean_official_pas_delta_vs_v7": float(np.mean(pas_delta[mask])),
                    "positive_official_pas_fraction": float(np.mean(pas_delta[mask] > 0.0)),
                    "negative_official_pas_fraction": float(np.mean(pas_delta[mask] < 0.0)),
                }
            )
    return rows


def nearest_peak_distance(peaks: list[e169.Peak], h: int, v: int) -> float:
    if not peaks:
        return float("inf")
    anchor = e169.Peak(h=h, v=v, value=0.0, mass=0.0)
    return min(e169.scaled_distance(peak, anchor)[0] for peak in peaks)


def event_repaired(
    event: dict[str, object],
    v7_map: np.ndarray,
    v12_map: np.ndarray,
    truth_map: np.ndarray,
) -> tuple[bool, float]:
    event_type = str(event["event_type"])
    if event_type == "spurious_peak":
        h, v = int(event["pred_h"]), int(event["pred_v"])
    else:
        h, v = int(event["truth_h"]), int(event["truth_v"])
    hi = e169.angle_indices(h, 1, e169.N_H)
    vi = e169.angle_indices(v, 1, e169.N_V)
    before = float(np.abs(v7_map[np.ix_(hi, vi)] - truth_map[np.ix_(hi, vi)]).sum())
    after = float(np.abs(v12_map[np.ix_(hi, vi)] - truth_map[np.ix_(hi, vi)]).sum())
    reduction = before - after
    v12_peaks = e169.extract_peaks(v12_map)
    if event_type == "missing_peak":
        repaired = nearest_peak_distance(v12_peaks, h, v) <= 1.0 and reduction > 0.0
    elif event_type == "spurious_peak":
        old_mass = float(event["pred_mass"])
        new_mass = e169.window_mass(v12_map, h, v)
        tolerance = max(e169.MASS_ABS_TOL, 0.05 * max(old_mass, new_mass))
        repaired = new_mass < old_mass - tolerance and reduction > 0.0
    elif event_type == "angle_shift":
        repaired = nearest_peak_distance(v12_peaks, h, v) < float(event["scaled_shift"]) and reduction > 0.0
    elif event_type == "power_mismatch":
        target = float(event["truth_mass"])
        before_mass = abs(float(event["pred_mass"]) - target)
        after_mass = abs(e169.window_mass(v12_map, h, v) - target)
        repaired = after_mass < before_mass and reduction > 0.0
    else:
        repaired = False
    return repaired, reduction


def repair_rows(
    v7_events: list[dict[str, object]],
    compact: dict[str, np.ndarray],
    truth: np.ndarray,
) -> list[dict[str, object]]:
    audited: list[dict[str, object]] = []
    for event in v7_events:
        if event["event_type"] not in ERROR_TYPES:
            continue
        point = int(event["oof_row"])
        ue = int(event["ue"])
        repaired, reduction = event_repaired(
            event, compact["V7"][point, ue], compact["V12"][point, ue], truth[point, ue]
        )
        audited.append({**event, "repaired": repaired, "local_l1_error_reduction": reduction})
    rows: list[dict[str, object]] = []
    for scope_type, scope in [("overall", "ALL"), ("difficulty", "easy"), ("difficulty", "medium"), ("difficulty", "hard")]:
        scoped = audited if scope == "ALL" else [event for event in audited if event["difficulty"] == scope]
        for error_type in ERROR_TYPES:
            selected = [event for event in scoped if event["event_type"] == error_type]
            total_weight = sum(float(event["error_weight"]) for event in selected)
            repaired_weight = sum(float(event["error_weight"]) for event in selected if bool(event["repaired"]))
            rows.append(
                {
                    "scope_type": scope_type,
                    "scope": scope,
                    "error_type": error_type,
                    "error_name": ERROR_NAMES[error_type],
                    "event_count": len(selected),
                    "repaired_count": sum(bool(event["repaired"]) for event in selected),
                    "count_repair_rate": float(np.mean([bool(event["repaired"]) for event in selected])) if selected else float("nan"),
                    "error_weight": total_weight,
                    "repaired_error_weight": repaired_weight,
                    "weight_repair_rate": repaired_weight / total_weight if total_weight > 0.0 else float("nan"),
                    "mean_local_l1_error_reduction": float(np.mean([float(event["local_l1_error_reduction"]) for event in selected])) if selected else float("nan"),
                }
            )
    return rows


def point_delta_rows(
    statistics: dict[str, PointMetricStatistics],
    hole: np.ndarray,
    difficulty: np.ndarray,
    depth: np.ndarray,
    site: np.ndarray,
    fold: np.ndarray,
    global_ids: np.ndarray,
    train_positions: np.ndarray,
) -> list[dict[str, object]]:
    v7 = statistics["V7"]
    v12 = statistics["V12"]
    v7_nmse = v7.squared_error / np.maximum(v7.truth_energy, TINY)
    v12_nmse = v12.squared_error / np.maximum(v12.truth_energy, TINY)
    v7_nmse_score = 1.0 / (1.0 + v7_nmse)
    v12_nmse_score = 1.0 / (1.0 + v12_nmse)
    v7_point_score = 0.4 * v7.pas + 0.4 * v7.pdp + 0.2 * v7_nmse_score
    v12_point_score = 0.4 * v12.pas + 0.4 * v12.pdp + 0.2 * v12_nmse_score
    rows = []
    for point in range(500):
        global_id = int(global_ids[point])
        rows.append(
            {
                "oof_row": point,
                "global_train_id": global_id,
                "x": float(train_positions[global_id, 0]),
                "y": float(train_positions[global_id, 1]),
                "site": int(site[point]),
                "fold": int(fold[point]),
                "pseudo_hole_id": str(hole[point]),
                "difficulty": str(difficulty[point]),
                "depth_bin": str(depth[point]),
                "v7_pas": float(v7.pas[point]),
                "v12_pas": float(v12.pas[point]),
                "delta_pas": float(v12.pas[point] - v7.pas[point]),
                "v7_pdp": float(v7.pdp[point]),
                "v12_pdp": float(v12.pdp[point]),
                "delta_pdp": float(v12.pdp[point] - v7.pdp[point]),
                "v7_point_nmse": float(v7_nmse[point]),
                "v12_point_nmse": float(v12_nmse[point]),
                "delta_point_nmse": float(v12_nmse[point] - v7_nmse[point]),
                "v7_diagnostic_point_score": float(v7_point_score[point]),
                "v12_diagnostic_point_score": float(v12_point_score[point]),
                "delta_diagnostic_point_score": float(v12_point_score[point] - v7_point_score[point]),
            }
        )
    return rows


def point_delta_summary_rows(point_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str, list[dict[str, object]]]] = [("overall", "ALL", point_rows)]
    for label in ("easy", "medium", "hard"):
        scopes.append(("difficulty", label, [row for row in point_rows if row["difficulty"] == label]))
    for label in DEPTH_ORDER:
        scopes.append(("depth", label, [row for row in point_rows if row["depth_bin"] == label]))
    for hole_id in sorted(EASY_HOLES + MEDIUM_HOLES + HARD_HOLES):
        scopes.append(("hole", hole_id, [row for row in point_rows if row["pseudo_hole_id"] == hole_id]))
    for scope_type, scope, selected in scopes:
        delta = np.asarray([float(row["delta_diagnostic_point_score"]) for row in selected], dtype=np.float64)
        pas = np.asarray([float(row["delta_pas"]) for row in selected], dtype=np.float64)
        pdp = np.asarray([float(row["delta_pdp"]) for row in selected], dtype=np.float64)
        rows.append(
            {
                "scope_type": scope_type,
                "scope": scope,
                "count": len(selected),
                "positive_point_fraction": float(np.mean(delta > 0.0)),
                "negative_point_fraction": float(np.mean(delta < 0.0)),
                "mean_diagnostic_point_score_delta": float(np.mean(delta)),
                "median_diagnostic_point_score_delta": float(np.median(delta)),
                "mean_pas_delta": float(np.mean(pas)),
                "mean_pdp_delta": float(np.mean(pdp)),
            }
        )
    return rows


def plot_overall(overall_rows: list[dict[str, object]]) -> Path:
    x = np.arange(len(overall_rows))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for key, label, color in (
        ("pas", "PAS", "#d1495b"),
        ("pdp", "PDP", "#30638e"),
        ("score", "总分", "#2a9d8f"),
    ):
        values = [float(row[key]) for row in overall_rows]
        ax.plot(x, values, marker="o", linewidth=2, label=label, color=color)
    ax.set_xticks(x, [row["version"] for row in overall_rows])
    ax.set_ylim(0.58, 0.79)
    ax.set_ylabel("官方指标")
    ax.set_title("V7-V12 fixed500整体官方指标")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = VIS_DIR / "01_V7到V12整体官方指标.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_hole_heatmap(hole_rows: list[dict[str, object]]) -> Path:
    holes = sorted({str(row["pseudo_hole_id"]) for row in hole_rows})
    values = np.asarray(
        [[next(float(row["score"]) for row in hole_rows if row["version"] == version and row["pseudo_hole_id"] == hole) for hole in holes] for version in VERSIONS]
    )
    fig, ax = plt.subplots(figsize=(13, 5.8))
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=float(values.min()), vmax=float(values.max()))
    ax.set_xticks(np.arange(len(holes)), holes, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(VERSIONS)), VERSIONS)
    ax.set_title("十个pseudo-hole逐版本官方总分")
    for i in range(len(VERSIONS)):
        for j in range(len(holes)):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white" if values[i, j] < np.median(values) else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="官方总分")
    fig.tight_layout()
    path = VIS_DIR / "02_十洞逐版本官方总分热图.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_difficulty(difficulty_rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = {"easy": "#2a9d8f", "medium": "#edae49", "hard": "#d1495b"}
    x = np.arange(len(VERSIONS))
    for label in ("easy", "medium", "hard"):
        values = [next(float(row["score"]) for row in difficulty_rows if row["version"] == version and row["difficulty"] == label) for version in VERSIONS]
        ax.plot(x, values, marker="o", linewidth=2, label=label, color=colors[label])
    ax.set_xticks(x, VERSIONS)
    ax.set_ylabel("官方总分")
    ax.set_title("easy / medium / hard洞的版本轨迹")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = VIS_DIR / "03_难度分层官方总分轨迹.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_hard_errors(error_rows: list[dict[str, object]]) -> Path:
    selected = {
        row["version"]: row
        for row in error_rows
        if row["scope_type"] == "difficulty" and row["scope"] == "hard"
    }
    x = np.arange(len(VERSIONS))
    bottom = np.zeros(len(VERSIONS), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = ("#d1495b", "#edae49", "#30638e", "#2a9d8f")
    for error_type, color in zip(ERROR_TYPES, colors, strict=True):
        values = np.asarray([float(selected[version][f"{error_type}_share"]) for version in VERSIONS])
        ax.bar(x, values, bottom=bottom, label=ERROR_NAMES[error_type], color=color)
        bottom += values
    ax.set_xticks(x, VERSIONS)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("困难洞PAS峰错误质量占比")
    ax.set_title("困难洞PAS峰错误构成如何随版本变化")
    ax.legend(ncol=4)
    fig.tight_layout()
    path = VIS_DIR / "04_困难洞PAS峰错误演化.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str, str]]) -> list[str]:
    lines = ["| " + " | ".join(label for _, label, _ in columns) + " |", "|" + "|".join("---" if fmt == "s" else "---:" for _, _, fmt in columns) + "|"]
    for row in rows:
        values = []
        for key, _, fmt in columns:
            value = row[key]
            if fmt == "s":
                values.append(str(value))
            elif fmt == ".1%":
                values.append(f"{float(value):.1%}")
            else:
                values.append(format(float(value), fmt))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(
    overall_rows: list[dict[str, object]],
    contribution: list[dict[str, object]],
    hole_rows: list[dict[str, object]],
    difficulty_rows: list[dict[str, object]],
    depth_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
    action_rows: list[dict[str, object]],
    repair: list[dict[str, object]],
    point_rows: list[dict[str, object]],
    point_summary: list[dict[str, object]],
    figures: list[Path],
) -> None:
    step = [row for row in contribution if row["comparison_type"] == "step"]
    hard_errors = [row for row in error_rows if row["scope_type"] == "difficulty" and row["scope"] == "hard"]
    overall_actions = [row for row in action_rows if row["difficulty"] == "ALL"]
    overall_repair = [row for row in repair if row["scope_type"] == "overall"]
    point_strata = [row for row in point_summary if row["scope_type"] in {"difficulty", "depth"}]
    top_fixed = sorted(point_rows, key=lambda row: float(row["delta_diagnostic_point_score"]), reverse=True)[:12]
    top_harmed = sorted(point_rows, key=lambda row: float(row["delta_diagnostic_point_score"]))[:12]
    lines = ["# V7到V12 fixed500完整审计", "", "所有PAS/PDP/NMSE/总分均由`code/common/official_metrics.py`在同一500点完整复信道上重算。逐点总分只用于排序修对/修坏点，不替代官方聚合总分。", "", "## 整体官方分", ""]
    lines += markdown_table(overall_rows, [("version", "版本", "s"), ("pas", "PAS", ".6f"), ("pdp", "PDP", ".6f"), ("nmse", "NMSE", ".6f"), ("score", "总分", ".6f"), ("delta_score_vs_v7", "相对V7", "+.6f"), ("online_score", "线上", ".6f")])
    lines += ["", "## 相邻版本贡献", ""]
    lines += markdown_table(step, [("base", "父版本", "s"), ("changed", "版本", "s"), ("delta_score", "总分增量", "+.6f"), ("pas_score_contribution", "PAS贡献", "+.6f"), ("pdp_score_contribution", "PDP贡献", "+.6f"), ("nmse_score_contribution", "NMSE贡献", "+.6f")])
    lines += ["", "## 十个pseudo-hole完整官方分", "", "完整四指标长表如下；每个版本和洞各一行。", ""]
    lines += markdown_table(hole_rows, [("version", "版本", "s"), ("pseudo_hole_id", "洞", "s"), ("difficulty", "难度", "s"), ("pas", "PAS", ".6f"), ("pdp", "PDP", ".6f"), ("nmse", "NMSE", ".6f"), ("score", "总分", ".6f"), ("delta_score_vs_parent", "相对父版", "+.6f"), ("delta_score_vs_v7", "相对V7", "+.6f")])
    lines += ["", "## easy / medium / hard", ""]
    lines += markdown_table(difficulty_rows, [("version", "版本", "s"), ("difficulty", "难度", "s"), ("pas", "PAS", ".6f"), ("pdp", "PDP", ".6f"), ("nmse", "NMSE", ".6f"), ("score", "总分", ".6f"), ("delta_score_vs_v7", "相对V7", "+.6f")])
    lines += ["", "## 洞边缘 / 中部 / 深处", ""]
    lines += markdown_table(depth_rows, [("version", "版本", "s"), ("depth_bin", "深度", "s"), ("pas", "PAS", ".6f"), ("pdp", "PDP", ".6f"), ("nmse", "NMSE", ".6f"), ("score", "总分", ".6f"), ("delta_score_vs_v7", "相对V7", "+.6f")])
    lines += ["", "## 困难洞PAS峰错误", ""]
    lines += markdown_table(hard_errors, [("version", "版本", "s"), ("error_weight_per_point_ue", "错误质量/点UE", ".6f"), ("missing_peak_share", "漏峰", ".1%"), ("spurious_peak_share", "假峰", ".1%"), ("angle_shift_share", "角度偏移", ".1%"), ("power_mismatch_share", "功率不匹配", ".1%")])
    lines += ["", "## PAS峰动作", "", "新增/删除/移动均按逐点逐UE的`16x8`compact PAS峰统计；总变差是实际归一化PAS能量移动量。", ""]
    lines += markdown_table(overall_actions, [("base", "基线", "s"), ("changed", "版本", "s"), ("mean_compact_pas_total_variation", "平均能量移动", ".6f"), ("deleted_count", "删峰", ".0f"), ("added_count", "加峰", ".0f"), ("moved_count", "移峰", ".0f"), ("increased_count", "增强", ".0f"), ("decreased_count", "减弱", ".0f"), ("beneficial_event_rate", "局部改善率", ".1%")])
    lines += ["", "## V12对V7四类错误的修复率", ""]
    lines += markdown_table(overall_repair, [("error_name", "V7错误", "s"), ("event_count", "事件数", ".0f"), ("count_repair_rate", "按事件修复率", ".1%"), ("weight_repair_rate", "按错误质量修复率", ".1%"), ("mean_local_l1_error_reduction", "局部L1减少", "+.6f")])
    lines += ["", "## V12逐点修对/修坏比例", ""]
    lines += markdown_table(point_strata, [("scope", "分层", "s"), ("count", "点数", ".0f"), ("positive_point_fraction", "修对比例", ".1%"), ("negative_point_fraction", "修坏比例", ".1%"), ("mean_diagnostic_point_score_delta", "平均诊断分变化", "+.6f")])
    lines += ["", "## V12最明显修对的点", ""]
    lines += markdown_table(top_fixed, [("oof_row", "OOF行", ".0f"), ("global_train_id", "训练ID", ".0f"), ("pseudo_hole_id", "洞", "s"), ("difficulty", "难度", "s"), ("depth_bin", "深度", "s"), ("delta_pas", "PAS变化", "+.6f"), ("delta_pdp", "PDP变化", "+.6f"), ("delta_diagnostic_point_score", "诊断分变化", "+.6f")])
    lines += ["", "## V12最明显修坏的点", ""]
    lines += markdown_table(top_harmed, [("oof_row", "OOF行", ".0f"), ("global_train_id", "训练ID", ".0f"), ("pseudo_hole_id", "洞", "s"), ("difficulty", "难度", "s"), ("depth_bin", "深度", "s"), ("delta_pas", "PAS变化", "+.6f"), ("delta_pdp", "PDP变化", "+.6f"), ("delta_diagnostic_point_score", "诊断分变化", "+.6f")])
    lines += ["", "## 图", ""]
    for path in figures:
        relative = Path("../../visualizations/复赛V7到V12完整洞审计") / path.name
        lines += [f"![{path.stem}]({relative.as_posix()})", ""]
    lines += ["## 边界", "", "本轮只做历史结果诊断，没有训练模型、生成新Sionna数据、修改V7-V12或生成提交包。", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


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
    print(f"E170 device={device} gpu={gpu_name}", flush=True)

    v8 = reconstruct_v8(device, args.batch_size, args.force_v8)
    channels = {version: validate_channel(path) for version, path in CHANNEL_PATHS.items()}
    channels["V8"] = v8
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    if len(query_ids) != 500 or len(np.unique(query_ids)) != 500:
        raise RuntimeError("canonical query IDs are invalid")
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    depth, global_ids = load_depth_labels()
    if not np.array_equal(global_ids, query_ids):
        raise RuntimeError("E167 point rows do not match canonical query order")
    hole, difficulty, site, fold, _ = build_point_labels(query_ids)

    statistics: dict[str, PointMetricStatistics] = {}
    overalls: dict[str, dict[str, float | int]] = {}
    identities: dict[str, dict[str, float]] = {}
    for version in VERSIONS:
        statistics[version], overalls[version] = score_channel(
            channels[version], truth, query_ids, args.batch_size
        )
        identities[version] = identity_errors(version, overalls[version])
        print(
            f"E170 {version}: PAS={overalls[version]['pas']:.9f} "
            f"PDP={overalls[version]['pdp']:.9f} NMSE={overalls[version]['nmse']:.9f} "
            f"score={overalls[version]['score']:.9f}",
            flush=True,
        )

    overall_rows, hole_rows, difficulty_rows, depth_rows = score_tables(
        statistics, overalls, hole, difficulty, depth
    )
    contribution = contribution_rows(overalls)
    compact, truth_compact = build_compact_maps(
        channels, truth, query_ids, device, args.fft_batch_size, args.rebuild_compact
    )
    representation = representation_gate(compact, truth_compact, statistics)

    events_by_version: dict[str, list[dict[str, object]]] = {}
    for version in VERSIONS:
        events_by_version[version] = build_error_events(
            version, compact[version], truth_compact, hole, difficulty, depth
        )
        print(f"E170 peak errors {version}: {len(events_by_version[version])}", flush=True)
    error_rows = error_summary_rows(events_by_version, hole, difficulty, depth)
    action_rows = action_summary_rows(compact, truth_compact, difficulty)
    correction = correction_rows(compact, truth_compact, statistics, hole, difficulty, depth)
    repair = repair_rows(events_by_version["V7"], compact, truth_compact)
    _, _, context = e167.build_holes()
    train_positions = np.asarray(context["train_positions"], dtype=np.float64)
    point_rows = point_delta_rows(
        statistics, hole, difficulty, depth, site, fold, global_ids, train_positions
    )
    point_summary = point_delta_summary_rows(point_rows)

    write_csv(OVERALL_CSV, overall_rows)
    write_csv(CONTRIBUTION_CSV, contribution)
    write_csv(HOLE_CSV, hole_rows)
    write_csv(DIFFICULTY_CSV, difficulty_rows)
    write_csv(DEPTH_CSV, depth_rows)
    write_csv(ERROR_CSV, error_rows)
    write_csv(ACTION_CSV, action_rows)
    write_csv(CORRECTION_CSV, correction)
    write_csv(REPAIR_CSV, repair)
    write_csv(POINT_CSV, point_rows)
    write_csv(POINT_SUMMARY_CSV, point_summary)

    figures = [
        plot_overall(overall_rows),
        plot_hole_heatmap(hole_rows),
        plot_difficulty(difficulty_rows),
        plot_hard_errors(error_rows),
    ]
    write_markdown(
        overall_rows,
        contribution,
        hole_rows,
        difficulty_rows,
        depth_rows,
        error_rows,
        action_rows,
        repair,
        point_rows,
        point_summary,
        figures,
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "device": str(device),
        "gpu": gpu_name,
        "identity_errors": identities,
        "representation_gate": representation,
        "overall": overall_rows,
        "score_contributions": contribution,
        "per_hole": hole_rows,
        "difficulty": difficulty_rows,
        "depth": depth_rows,
        "pas_peak_error_summary": error_rows,
        "pas_peak_actions": action_rows,
        "pas_correction": correction,
        "v12_repair_of_v7_errors": repair,
        "v12_point_delta_summary": point_summary,
        "top_repaired_points": sorted(point_rows, key=lambda row: float(row["delta_diagnostic_point_score"]), reverse=True)[:20],
        "top_harmed_points": sorted(point_rows, key=lambda row: float(row["delta_diagnostic_point_score"]))[:20],
        "outputs": [
            str(PATHS.relative(path))
            for path in (
                REPORT_MD,
                OVERALL_CSV,
                CONTRIBUTION_CSV,
                HOLE_CSV,
                DIFFICULTY_CSV,
                DEPTH_CSV,
                ERROR_CSV,
                ACTION_CSV,
                CORRECTION_CSV,
                REPAIR_CSV,
                POINT_CSV,
                POINT_SUMMARY_CSV,
                *figures,
            )
        ],
        "seconds": time.perf_counter() - started,
    }
    REPORT_JSON.write_text(
        json.dumps(json_compatible(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"E170 complete in {report['seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
