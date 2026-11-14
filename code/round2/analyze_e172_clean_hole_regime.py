#!/usr/bin/env python3
"""Audit whether Clean observables predict whole-hole V7 correction regimes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import analyze_e167_clean_test_hole_difficulty as e167
import analyze_e169_hard_hole_pas_peak_precursors as e169
from common.official_metrics import PointMetricStatistics, aggregate_statistics
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E172-clean-hole-regime"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e172_clean_hole_regime"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e172_clean_hole_regime"
VIS_DIR = PATHS.analysis / "visualizations" / "复赛Clean洞级修正Regime审计"

FIELD_CACHE = ARTIFACT_DIR / "full4000_v7_core_clean_fields.npz"
REPORT_JSON = ANALYSIS_DIR / "clean_hole_regime.json"
REPORT_MD = ANALYSIS_DIR / "峰级规律跨洞失效后洞级规律是否稳定.md"
PROXY_CSV = ANALYSIS_DIR / "完整V7与严格V7core十洞代理门.csv"
HOLE_CSV = ANALYSIS_DIR / "45个Clean_pseudo洞几何标签与官方分.csv"
FEATURE_CSV = ANALYSIS_DIR / "Clean洞级特征.csv"
MATCH_CSV = ANALYSIS_DIR / "真实Test洞与扩展pseudo洞匹配.csv"
MODEL_CSV = ANALYSIS_DIR / "整外折洞级分类结果.csv"
FOLD_CSV = ANALYSIS_DIR / "整外折逐折分类结果.csv"
PREDICTION_CSV = ANALYSIS_DIR / "严格整外折逐洞预测.csv"
PERMUTATION_CSV = ANALYSIS_DIR / "site内标签置换对照.csv"
FEATURE_EFFECT_CSV = ANALYSIS_DIR / "压峰洞与不动洞Clean特征差异.csv"
CLUSTER_CSV = ANALYSIS_DIR / "洞级错误构成聚类.csv"

FULL_CORE = (
    PATHS.artifacts
    / "round2"
    / "e160_full4000_v7_supervision"
    / "full4000_v7_core.npy"
)
CANONICAL_V7 = e167.V7_COMPLETE
TRAIN_CHANNEL = e167.TRAIN_CHANNEL
FOLD_IDS = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)
TRAIN_COMPACT = e169.TRAIN_COMPACT
CANONICAL_COMPACT = e169.V7_COMPACT
MAP_FEATURES = e167.MAP_FEATURES
SPECTRAL_DESCRIPTORS = e167.SPECTRAL_DESCRIPTORS

N_UE = 4
N_POL = 2
N_H = 16
N_V = 8
N_SC = 192
TINY = 1e-30
SEED = 20_260_811_172
BOOTSTRAPS = 200
PERMUTATIONS = 500
COMPONENT_RADIUS_M = 6.0
MIN_HOLE_POINTS = 20
MAX_HOLE_POINTS = 60
PAS_NO_ACTION = 0.72
DOMINANT_SHARE = 0.40
DOMINANT_MARGIN = 0.10

ERROR_TYPES = (
    "missing_peak",
    "spurious_peak",
    "angle_shift",
    "power_mismatch",
)
REGIME_ORDER = (
    "no_action",
    "suppress",
    "shift",
    "redistribute",
    "missing",
    "mixed",
)
REGIME_CN = {
    "no_action": "基本不用动",
    "suppress": "整体偏向压峰",
    "shift": "峰位置问题",
    "redistribute": "功率重新分配",
    "missing": "漏峰严重",
    "mixed": "混合型",
}
ERROR_TO_REGIME = {
    "spurious_peak": "suppress",
    "angle_shift": "shift",
    "power_mismatch": "redistribute",
    "missing_peak": "missing",
}
MAP_FIELDS = e167.MAP_MATCH_FEATURES
MODEL_NAMES = ("logistic", "decision_tree", "extra_trees")


@dataclass(frozen=True)
class HoleProfile:
    label: str
    shares: np.ndarray
    point_error_mass: np.ndarray
    bootstrap_stability: float
    bootstrap_label_counts: dict[str, int]
    bootstrap_share_q05: np.ndarray
    bootstrap_share_q95: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force-fields", action="store_true")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
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
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def entropy_rows(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = array.sum(axis=1, keepdims=True)
    probability = np.divide(array, np.maximum(total, TINY), where=total > 0.0)
    log_probability = np.zeros_like(probability)
    positive = probability > 0.0
    log_probability[positive] = np.log(probability[positive])
    return -np.sum(probability * log_probability, axis=1) / math.log(array.shape[1])


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return 0.0 if np.linalg.norm(a) == np.linalg.norm(b) else 1.0
    return float(1.0 - np.dot(a, b) / denominator)


def torch_cosine(prediction: torch.Tensor, truth: torch.Tensor, axis: int) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=axis, dtype=torch.float64)
    pred_norm = prediction.square().sum(dim=axis, dtype=torch.float64)
    truth_norm = truth.square().sum(dim=axis, dtype=torch.float64)
    denominator = (pred_norm * truth_norm).sqrt()
    denominator = torch.where(denominator == 0.0, torch.full_like(denominator, 1e-12), denominator)
    return numerator / denominator


@torch.inference_mode()
def build_full_fields(device: torch.device, batch_size: int, force: bool) -> dict[str, np.ndarray]:
    expected = {
        "compact_pas": (4000, N_UE, N_H, N_V),
        "global_pdp": (4000, N_SC),
        "pas": (4000,),
        "pdp": (4000,),
        "squared_error": (4000,),
        "truth_energy": (4000,),
        "total_power": (4000,),
    }
    if FIELD_CACHE.is_file() and not force:
        cache = np.load(FIELD_CACHE, allow_pickle=False)
        if all(key in cache and cache[key].shape == shape for key, shape in expected.items()):
            return {key: cache[key] for key in expected}

    prediction = np.load(FULL_CORE, mmap_mode="r")
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    if prediction.shape != truth.shape or prediction.shape != (4000, 256, 4, 192):
        raise ValueError(f"unexpected full4000 channel shapes: {prediction.shape} {truth.shape}")
    output = {
        "compact_pas": np.empty(expected["compact_pas"], dtype=np.float32),
        "global_pdp": np.empty(expected["global_pdp"], dtype=np.float32),
        "pas": np.empty(4000, dtype=np.float64),
        "pdp": np.empty(4000, dtype=np.float64),
        "squared_error": np.empty(4000, dtype=np.float64),
        "truth_energy": np.empty(4000, dtype=np.float64),
        "total_power": np.empty(4000, dtype=np.float64),
    }
    for start in range(0, len(prediction), batch_size):
        end = min(start + batch_size, len(prediction))
        pred = torch.as_tensor(
            np.array(prediction[start:end], dtype=np.complex64, copy=True), device=device
        )
        target = torch.as_tensor(
            np.array(truth[start:end], dtype=np.complex64, copy=True), device=device
        )
        batch = end - start
        pred_shaped = pred.reshape(batch, N_POL, N_H, N_V, N_UE, N_SC)
        truth_shaped = target.reshape(batch, N_POL, N_H, N_V, N_UE, N_SC)
        pred_pas = torch.fft.fft2(pred_shaped, dim=(2, 3)).abs().square()
        truth_pas = torch.fft.fft2(truth_shaped, dim=(2, 3)).abs().square()
        pred_pas_flat = pred_pas.reshape(batch, 256, N_UE, N_SC)
        truth_pas_flat = truth_pas.reshape(batch, 256, N_UE, N_SC)
        output["pas"][start:end] = (
            torch_cosine(pred_pas_flat, truth_pas_flat, 1)
            .mean(dim=(1, 2), dtype=torch.float64)
            .cpu()
            .numpy()
        )

        compact = pred_pas.sum(dim=1).mean(dim=-1).permute(0, 3, 1, 2)
        compact_total = compact.sum(dim=(2, 3), keepdim=True)
        compact = torch.where(
            compact_total > 0.0,
            compact / compact_total.clamp_min(TINY),
            torch.zeros_like(compact),
        )
        output["compact_pas"][start:end] = compact.float().cpu().numpy()

        pred_pdp = torch.fft.ifft(pred, dim=-1).abs().square()
        truth_pdp = torch.fft.ifft(target, dim=-1).abs().square()
        output["pdp"][start:end] = (
            torch_cosine(pred_pdp, truth_pdp, 3)
            .mean(dim=(1, 2), dtype=torch.float64)
            .cpu()
            .numpy()
        )
        global_pdp = pred_pdp.mean(dim=(1, 2))
        global_total = global_pdp.sum(dim=1, keepdim=True)
        global_pdp = torch.where(
            global_total > 0.0,
            global_pdp / global_total.clamp_min(TINY),
            torch.zeros_like(global_pdp),
        )
        output["global_pdp"][start:end] = global_pdp.float().cpu().numpy()
        output["squared_error"][start:end] = (
            (pred - target).abs().square().sum(dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
        )
        output["truth_energy"][start:end] = (
            target.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
        )
        output["total_power"][start:end] = (
            pred.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
        )
        if end == len(prediction) or end % 256 == 0:
            print(f"E172 full4000 Clean fields: {end}/{len(prediction)}", flush=True)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIELD_CACHE, **output)
    return output


def load_complete_point_metrics() -> dict[str, np.ndarray]:
    rows: list[dict[str, str] | None] = [None] * 500
    with e167.POINT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["oof_row"])] = row
    if any(row is None for row in rows):
        raise RuntimeError("E167 point metric table does not cover canonical 500 points")
    typed = [row for row in rows if row is not None]
    return {
        "pas": np.asarray([float(row["pas"]) for row in typed], dtype=np.float64),
        "pdp": np.asarray([float(row["pdp"]) for row in typed], dtype=np.float64),
        "squared_error": np.asarray(
            [float(row["squared_error"]) for row in typed], dtype=np.float64
        ),
        "truth_energy": np.asarray(
            [float(row["truth_energy"]) for row in typed], dtype=np.float64
        ),
    }


def statistics_from_fields(fields: dict[str, np.ndarray], ids: np.ndarray) -> PointMetricStatistics:
    selected = np.asarray(ids, dtype=np.int64)
    return PointMetricStatistics(
        pas=np.asarray(fields["pas"])[selected],
        pdp=np.asarray(fields["pdp"])[selected],
        squared_error=np.asarray(fields["squared_error"])[selected],
        truth_energy=np.asarray(fields["truth_energy"])[selected],
    )


def regime_label(pas: float, shares: np.ndarray) -> str:
    if pas >= PAS_NO_ACTION:
        return "no_action"
    order = np.argsort(shares)[::-1]
    if shares[order[0]] >= DOMINANT_SHARE and shares[order[0]] - shares[order[1]] >= DOMINANT_MARGIN:
        return ERROR_TO_REGIME[ERROR_TYPES[int(order[0])]]
    return "mixed"


def point_error_masses(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if prediction.shape != truth.shape:
        raise ValueError(f"compact PAS mismatch: {prediction.shape} != {truth.shape}")
    output = np.zeros((len(prediction), len(ERROR_TYPES)), dtype=np.float64)
    index = {name: position for position, name in enumerate(ERROR_TYPES)}
    for point in range(len(prediction)):
        for ue in range(N_UE):
            for event in e169.match_peaks(prediction[point, ue], truth[point, ue]):
                event_type = str(event["event_type"])
                if event_type in index:
                    output[point, index[event_type]] += float(event["error_weight"])
    return output


def profile_hole(
    prediction: np.ndarray,
    truth: np.ndarray,
    point_pas: np.ndarray,
    rng: np.random.Generator,
    bootstraps: int,
) -> HoleProfile:
    masses = point_error_masses(prediction, truth)
    totals = masses.sum(axis=0)
    shares = totals / max(float(totals.sum()), TINY)
    label = regime_label(float(np.mean(point_pas)), shares)
    if bootstraps <= 0:
        return HoleProfile(
            label=label,
            shares=shares,
            point_error_mass=masses,
            bootstrap_stability=float("nan"),
            bootstrap_label_counts={},
            bootstrap_share_q05=np.full(4, np.nan),
            bootstrap_share_q95=np.full(4, np.nan),
        )
    bootstrap_shares = np.empty((bootstraps, len(ERROR_TYPES)), dtype=np.float64)
    labels: list[str] = []
    for repeat in range(bootstraps):
        sample = rng.integers(0, len(masses), size=len(masses))
        total = masses[sample].sum(axis=0)
        local_share = total / max(float(total.sum()), TINY)
        bootstrap_shares[repeat] = local_share
        labels.append(regime_label(float(np.mean(point_pas[sample])), local_share))
    counts = {name: labels.count(name) for name in REGIME_ORDER}
    return HoleProfile(
        label=label,
        shares=shares,
        point_error_mass=masses,
        bootstrap_stability=float(np.mean(np.asarray(labels) == label)),
        bootstrap_label_counts=counts,
        bootstrap_share_q05=np.quantile(bootstrap_shares, 0.05, axis=0),
        bootstrap_share_q95=np.quantile(bootstrap_shares, 0.95, axis=0),
    )


def share_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 1.0 if np.linalg.norm(left) == np.linalg.norm(right) else 0.0
    return float(np.dot(left, right) / denominator)


def canonical_proxy_gate(
    fields: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[dict[str, object], list[dict[str, object]]]:
    _, holes, _ = e167.build_holes()
    metadata = np.load(e167.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    global_to_oof = np.full(4000, -1, dtype=np.int64)
    global_to_oof[query_ids] = np.arange(500, dtype=np.int64)
    complete_metrics = load_complete_point_metrics()
    complete_stats = PointMetricStatistics(**complete_metrics)
    complete_overall = aggregate_statistics(complete_stats)
    core_stats = statistics_from_fields(fields, query_ids)
    core_overall = aggregate_statistics(core_stats)
    truth_compact = np.load(TRAIN_COMPACT, mmap_mode="r")
    complete_compact = np.load(CANONICAL_COMPACT, mmap_mode="r")
    core_compact = np.asarray(fields["compact_pas"])

    rows: list[dict[str, object]] = []
    agreements: list[float] = []
    cosines: list[float] = []
    for hole in holes:
        oof_rows = global_to_oof[hole.point_ids]
        complete_profile = profile_hole(
            np.asarray(complete_compact[oof_rows]),
            np.asarray(truth_compact[hole.point_ids]),
            complete_metrics["pas"][oof_rows],
            rng,
            0,
        )
        core_profile = profile_hole(
            core_compact[hole.point_ids],
            np.asarray(truth_compact[hole.point_ids]),
            np.asarray(fields["pas"])[hole.point_ids],
            rng,
            0,
        )
        cosine = share_cosine(complete_profile.shares, core_profile.shares)
        agreement = float(complete_profile.label == core_profile.label)
        agreements.append(agreement)
        cosines.append(cosine)
        row: dict[str, object] = {
            "hole_id": hole.hole_id,
            "site": hole.site,
            "fold": hole.ordinal,
            "point_count": len(hole.point_ids),
            "complete_pas": float(np.mean(complete_metrics["pas"][oof_rows])),
            "core_pas": float(np.mean(np.asarray(fields["pas"])[hole.point_ids])),
            "complete_regime": complete_profile.label,
            "core_regime": core_profile.label,
            "regime_agree": bool(agreement),
            "error_share_cosine": cosine,
        }
        for index, error_type in enumerate(ERROR_TYPES):
            row[f"complete_{error_type}_share"] = float(complete_profile.shares[index])
            row[f"core_{error_type}_share"] = float(core_profile.shares[index])
        rows.append(row)
    agreement_rate = float(np.mean(agreements))
    mean_cosine = float(np.mean(cosines))
    passed = agreement_rate >= 0.70 and mean_cosine >= 0.85
    return {
        "passed": passed,
        "regime_agreement": agreement_rate,
        "mean_error_share_cosine": mean_cosine,
        "required_regime_agreement": 0.70,
        "required_mean_error_share_cosine": 0.85,
        "complete_overall": complete_overall,
        "core_same500_overall": core_overall,
        "core_minus_complete": {
            key: float(core_overall[key]) - float(complete_overall[key])
            for key in ("pas", "pdp", "nmse", "score")
        },
    }, rows


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def points_inside_hull(hull: ConvexHull, points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    return np.all(points @ hull.equations[:, :-1].T + hull.equations[:, -1] <= tolerance, axis=1)


def build_expanded_holes() -> tuple[list[e167.Hole], dict[str, object]]:
    _, train_positions, _, train_site, _ = load_positions()
    fold_id = np.load(FOLD_IDS).astype(np.int8)
    map_cache = np.load(MAP_FEATURES, allow_pickle=False)
    train_map = map_cache["train_features"].astype(np.float32)
    holes: list[e167.Hole] = []
    rejected = {"size": 0, "collinear": 0, "contaminated": 0}
    component_counts: dict[str, int] = {}
    for fold in range(5):
        for site in (0, 1):
            ids = np.flatnonzero((fold_id == fold) & (train_site == site)).astype(np.int64)
            xy = train_positions[ids, :2]
            union = UnionFind(len(ids))
            for left, right in cKDTree(xy).query_pairs(COMPONENT_RADIUS_M):
                union.union(int(left), int(right))
            groups: dict[int, list[int]] = {}
            for local in range(len(ids)):
                groups.setdefault(union.find(local), []).append(local)
            retained = 0
            for local_group in groups.values():
                point_ids = np.sort(ids[np.asarray(local_group, dtype=np.int64)])
                if not MIN_HOLE_POINTS <= len(point_ids) <= MAX_HOLE_POINTS:
                    rejected["size"] += 1
                    continue
                point_xy = train_positions[point_ids, :2]
                try:
                    hull = ConvexHull(point_xy)
                except Exception:
                    rejected["collinear"] += 1
                    continue
                if float(hull.volume) <= 1e-6:
                    rejected["collinear"] += 1
                    continue
                other_ids = np.flatnonzero((fold_id != fold) & (train_site == site)).astype(np.int64)
                if np.any(points_inside_hull(hull, train_positions[other_ids, :2])):
                    rejected["contaminated"] += 1
                    continue
                fit_ids = other_ids
                ignored_ids = np.setdiff1d(ids, point_ids, assume_unique=True)
                hole_id = f"E_S{site}F{fold}C{retained:02d}"
                holes.append(
                    e167.Hole(
                        hole_id=hole_id,
                        kind="expanded_clean_pseudo",
                        site=site,
                        ordinal=fold,
                        point_ids=point_ids,
                        xy=point_xy.astype(np.float64),
                        fit_ids=fit_ids,
                        ignored_ids=ignored_ids,
                        map_features=train_map[point_ids],
                    )
                )
                retained += 1
            component_counts[f"fold{fold}_site{site}"] = retained
    return holes, {
        "hole_count": len(holes),
        "component_counts": component_counts,
        "rejected": rejected,
        "point_count_min": min(len(hole.point_ids) for hole in holes) if holes else 0,
        "point_count_max": max(len(hole.point_ids) for hole in holes) if holes else 0,
    }


def profile_expanded_holes(
    holes: list[e167.Hole], fields: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[list[dict[str, object]], dict[str, HoleProfile]]:
    truth_compact = np.load(TRAIN_COMPACT, mmap_mode="r")
    core_compact = np.asarray(fields["compact_pas"])
    rows: list[dict[str, object]] = []
    profiles: dict[str, HoleProfile] = {}
    for hole in holes:
        ids = hole.point_ids
        official = aggregate_statistics(statistics_from_fields(fields, ids))
        profile = profile_hole(
            core_compact[ids],
            np.asarray(truth_compact[ids]),
            np.asarray(fields["pas"])[ids],
            rng,
            BOOTSTRAPS,
        )
        profiles[hole.hole_id] = profile
        row: dict[str, object] = {
            "hole_id": hole.hole_id,
            "site": hole.site,
            "broad_fold": hole.ordinal,
            "point_count": len(ids),
            "regime": profile.label,
            "regime_cn": REGIME_CN[profile.label],
            "bootstrap_stability": profile.bootstrap_stability,
            **official,
        }
        for index, error_type in enumerate(ERROR_TYPES):
            row[f"{error_type}_share"] = float(profile.shares[index])
            row[f"{error_type}_share_q05"] = float(profile.bootstrap_share_q05[index])
            row[f"{error_type}_share_q95"] = float(profile.bootstrap_share_q95[index])
        for label in REGIME_ORDER:
            row[f"bootstrap_{label}_count"] = profile.bootstrap_label_counts[label]
        rows.append(row)
        print(
            f"E172 regime {hole.hole_id}: {profile.label} "
            f"PAS={official['pas']:.6f} stability={profile.bootstrap_stability:.3f}",
            flush=True,
        )
    return rows, profiles


def normalized_rows(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = array.sum(axis=1, keepdims=True)
    return np.divide(array, np.maximum(total, TINY), where=total > 0.0)


def add_feature(row: dict[str, object], name: str, value: float) -> None:
    row[f"feature_{name}"] = float(value)


def extract_hole_features(
    holes: list[e167.Hole], fields: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    setup, train_positions, _, train_site, _ = load_positions()
    map_cache = np.load(MAP_FEATURES, allow_pickle=False)
    map_names = [str(item) for item in map_cache["feature_names"].tolist()]
    map_index = {name: index for index, name in enumerate(map_names)}
    train_map = map_cache["train_features"].astype(np.float32)
    descriptors = np.load(SPECTRAL_DESCRIPTORS, allow_pickle=False)
    train_pas = normalized_rows(descriptors["global_pas"])
    train_pdp = normalized_rows(descriptors["global_pdp"])
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    core_pas = np.asarray(fields["compact_pas"], dtype=np.float32)
    core_pdp = np.asarray(fields["global_pdp"], dtype=np.float32)
    core_power = np.asarray(fields["total_power"], dtype=np.float64)
    rows: list[dict[str, object]] = []

    for hole in holes:
        ids = hole.point_ids
        xy = train_positions[ids, :2].astype(np.float64)
        fit_xy = train_positions[hole.fit_ids, :2].astype(np.float64)
        visible = e167.visibility_arrays(xy, fit_xy, 16)
        minimum = np.min(xy, axis=0)
        maximum = np.max(xy, axis=0)
        width, height = maximum - minimum
        major, minor, aspect = e167.principal_axes(xy)
        center = np.mean(xy, axis=0)
        bs_delta = xy - setup.bs_positions[hole.site, :2]
        bs_distance = np.linalg.norm(bs_delta, axis=1)
        azimuth = np.arctan2(bs_delta[:, 1], bs_delta[:, 0])

        donor_count = min(32, len(hole.fit_ids))
        donor_distance, donor_local = cKDTree(fit_xy).query(center, k=donor_count)
        donor_distance = np.atleast_1d(donor_distance).astype(np.float64)
        donor_ids = hole.fit_ids[np.atleast_1d(donor_local).astype(np.int64)]
        donor_xy = train_positions[donor_ids, :2]
        donor_pas = train_pas[donor_ids]
        donor_pdp = train_pdp[donor_ids]
        side_pas, side_pdp = e167.two_side_spectral_difference(
            center, donor_xy, donor_pas, donor_pdp, minor
        )

        local_pas = core_pas[ids].reshape(len(ids), N_UE, -1)
        local_pas_mean_ue = normalized_rows(local_pas.mean(axis=1))
        local_pdp = normalized_rows(core_pdp[ids])
        pas_flat = local_pas.reshape(len(ids) * N_UE, -1)
        pas_sorted = np.sort(pas_flat, axis=1)[:, ::-1]
        pdp_sorted = np.sort(local_pdp, axis=1)[:, ::-1]
        peak_counts = np.asarray(
            [
                len(e169.extract_peaks(core_pas[global_id, ue]))
                for global_id in ids
                for ue in range(N_UE)
            ],
            dtype=np.float64,
        )
        log_power = np.log10(np.maximum(core_power[ids], 1e-30))
        neighbor_pas_mean = np.mean(donor_pas, axis=0)
        neighbor_pdp_mean = np.mean(donor_pdp, axis=0)
        v7_pas_mean = np.mean(local_pas_mean_ue, axis=0)
        v7_pdp_mean = np.mean(local_pdp, axis=0)

        row: dict[str, object] = {
            "hole_id": hole.hole_id,
            "site": hole.site,
            "broad_fold": hole.ordinal,
        }
        add_feature(row, "point_count", len(ids))
        add_feature(row, "width_m", width)
        add_feature(row, "height_m", height)
        add_feature(row, "bbox_area_m2", width * height)
        add_feature(row, "convex_hull_area_m2", e167.convex_hull_area(xy))
        add_feature(row, "pca_aspect_ratio", aspect)
        add_feature(row, "major_axis_x", major[0])
        add_feature(row, "major_axis_y", major[1])
        nearest = visible["distance"][:, 0]
        add_feature(row, "nearest_fit_mean_m", np.mean(nearest))
        add_feature(row, "nearest_fit_median_m", np.median(nearest))
        add_feature(row, "nearest_fit_p90_m", np.quantile(nearest, 0.90))
        add_feature(row, "nearest_fit_max_m", np.max(nearest))
        add_feature(row, "knn4_median_m", np.median(visible["distance"][:, 3]))
        add_feature(row, "knn8_median_m", np.median(visible["distance"][:, 7]))
        add_feature(row, "angular_gap_median_deg", np.median(visible["max_gap_deg"]))
        add_feature(row, "angular_gap_p90_deg", np.quantile(visible["max_gap_deg"], 0.90))
        add_feature(row, "one_sidedness_median", np.median(visible["one_sidedness"]))
        add_feature(row, "quadrants_median", np.median(visible["quadrants"]))
        add_feature(row, "bs_distance_mean_m", np.mean(bs_distance))
        add_feature(row, "bs_distance_range_m", np.ptp(bs_distance))
        add_feature(row, "bs_azimuth_sin_mean", np.mean(np.sin(azimuth)))
        add_feature(row, "bs_azimuth_cos_mean", np.mean(np.cos(azimuth)))
        for name in MAP_FIELDS:
            values = train_map[ids, map_index[name]].astype(np.float64)
            finite = values[np.isfinite(values)]
            add_feature(row, f"map_{name}_mean", np.mean(finite) if len(finite) else np.nan)
            add_feature(row, f"map_{name}_std", np.std(finite) if len(finite) else np.nan)

        add_feature(row, "neighbor_pas_dispersion", e167.cosine_dispersion(donor_pas))
        add_feature(row, "neighbor_pdp_dispersion", e167.cosine_dispersion(donor_pdp))
        add_feature(row, "two_side_pas_difference", side_pas)
        add_feature(row, "two_side_pdp_difference", side_pdp)
        add_feature(row, "neighbor_distance_mean_m", np.mean(donor_distance))
        add_feature(row, "neighbor_distance_max_m", np.max(donor_distance))
        h_take = min(16, len(donor_ids))
        add_feature(
            row,
            "neighbor_complex_h_variation_per_m",
            e167.complex_h_variation(train_h, donor_ids[:h_take], donor_xy[:h_take]),
        )

        add_feature(row, "v7_pas_top1_mean", np.mean(pas_sorted[:, 0]))
        add_feature(row, "v7_pas_top4_mass_mean", np.mean(pas_sorted[:, :4].sum(axis=1)))
        add_feature(row, "v7_pas_entropy_mean", np.mean(entropy_rows(pas_flat)))
        add_feature(row, "v7_pas_peak_count_mean", np.mean(peak_counts))
        add_feature(row, "v7_pas_peak_count_std", np.std(peak_counts))
        add_feature(row, "v7_pas_spatial_dispersion", e167.cosine_dispersion(local_pas_mean_ue))
        add_feature(row, "v7_pdp_top1_mean", np.mean(pdp_sorted[:, 0]))
        add_feature(row, "v7_pdp_top8_mass_mean", np.mean(pdp_sorted[:, :8].sum(axis=1)))
        add_feature(row, "v7_pdp_entropy_mean", np.mean(entropy_rows(local_pdp)))
        add_feature(row, "v7_pdp_spatial_dispersion", e167.cosine_dispersion(local_pdp))
        add_feature(row, "v7_to_neighbor_pas_distance", cosine_distance(v7_pas_mean, neighbor_pas_mean))
        add_feature(row, "v7_to_neighbor_pdp_distance", cosine_distance(v7_pdp_mean, neighbor_pdp_mean))
        add_feature(row, "v7_log_power_mean", np.mean(log_power))
        add_feature(row, "v7_log_power_std", np.std(log_power))
        add_feature(row, "v7_log_power_iqr", np.quantile(log_power, 0.75) - np.quantile(log_power, 0.25))
        add_feature(row, "v7_zero_fraction", np.mean(core_power[ids] <= 1e-30))
        rows.append(row)
        print(f"E172 Clean hole features {hole.hole_id}", flush=True)
    return rows


def attach_rows(
    holes: list[e167.Hole],
    regime_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    context: dict[str, object],
) -> list[dict[str, object]]:
    regime = {str(row["hole_id"]): row for row in regime_rows}
    features = {str(row["hole_id"]): row for row in feature_rows}
    map_names = np.asarray(context["map_names"])
    output: list[dict[str, object]] = []
    for hole in holes:
        geometry = e167.hole_summary(
            hole,
            np.asarray(context["train_positions"]),
            context["setup"].bs_positions,
            map_names,
        )
        output.append({**geometry, **regime[hole.hole_id], **features[hole.hole_id]})
    return output


def real_hole_matches(
    real_holes: list[e167.Hole],
    expanded_rows: list[dict[str, object]],
    context: dict[str, object],
) -> list[dict[str, object]]:
    real_rows = [
        e167.hole_summary(
            hole,
            np.asarray(context["train_positions"]),
            context["setup"].bs_positions,
            np.asarray(context["map_names"]),
        )
        for hole in real_holes
    ]
    combined = real_rows + expanded_rows
    vectors, groups = e167.matching_vectors(combined)
    real_vector = vectors[: len(real_rows)]
    expanded_vector = vectors[len(real_rows) :]
    rows: list[dict[str, object]] = []
    for real_index, real in enumerate(real_rows):
        candidates = []
        for pseudo_index, pseudo in enumerate(expanded_rows):
            if int(real["site"]) != int(pseudo["site"]):
                continue
            cost = e167.matching_cost(real_vector[real_index], expanded_vector[pseudo_index], groups)
            candidates.append((cost, pseudo_index))
        for rank, (cost, pseudo_index) in enumerate(sorted(candidates)[:3], start=1):
            pseudo = expanded_rows[pseudo_index]
            rows.append(
                {
                    "real_hole_id": real["hole_id"],
                    "site": real["site"],
                    "rank": rank,
                    "pseudo_hole_id": pseudo["hole_id"],
                    "match_cost": cost,
                    "real_point_count": real["point_count"],
                    "pseudo_point_count": pseudo["point_count"],
                    "real_nearest_train_median_m": real["nearest_train_median_m"],
                    "pseudo_nearest_train_median_m": pseudo["nearest_train_median_m"],
                    "pseudo_regime": pseudo["regime"],
                    "pseudo_pas": pseudo["pas"],
                    "pseudo_score": pseudo["score"],
                }
            )
    return rows


def model_factory(name: str, seed: int):
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if name == "decision_tree":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if name == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=128,
                min_samples_leaf=2,
                class_weight="balanced",
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    raise ValueError(name)


def cross_validated_predictions(
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    binary: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    prediction = np.empty(len(labels), dtype=labels.dtype)
    probability = np.full(len(labels), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    for group in np.unique(groups):
        train = groups != group
        test = groups == group
        train_classes = np.unique(labels[train])
        if len(train_classes) < 2:
            prediction[test] = train_classes[0]
            if binary:
                probability[test] = float(train_classes[0] == "suppress")
        else:
            model = model_factory(model_name, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x[train], labels[train])
            prediction[test] = model.predict(x[test])
            if binary:
                classes = list(model.classes_)
                probability[test] = model.predict_proba(x[test])[:, classes.index("suppress")]
        row: dict[str, object] = {
            "model": model_name,
            "task": "binary" if binary else "multiclass",
            "held_group": int(group),
            "test_count": int(np.sum(test)),
            "test_classes": ",".join(sorted(np.unique(labels[test]).tolist())),
            "accuracy": float(np.mean(prediction[test] == labels[test])),
            "balanced_accuracy": float(balanced_accuracy_score(labels[test], prediction[test])),
            "macro_f1": float(f1_score(labels[test], prediction[test], average="macro", zero_division=0)),
        }
        if binary:
            y_true = labels[test] == "suppress"
            y_pred = prediction[test] == "suppress"
            row["precision_suppress"] = float(precision_score(y_true, y_pred, zero_division=0))
            row["recall_suppress"] = float(recall_score(y_true, y_pred, zero_division=0))
            row["auc"] = (
                float(roc_auc_score(y_true, probability[test]))
                if len(np.unique(y_true)) == 2
                else float("nan")
            )
            row["direction_positive"] = bool(
                len(np.unique(y_true)) == 2 and float(row["balanced_accuracy"]) > 0.5
            )
        fold_rows.append(row)
    return prediction, probability, fold_rows


def evaluate_predictions(
    truth: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    binary: bool,
) -> dict[str, float]:
    output = {
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }
    if binary:
        y_true = truth == "suppress"
        y_pred = prediction == "suppress"
        output.update(
            {
                "auc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else float("nan"),
                "precision_suppress": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall_suppress": float(recall_score(y_true, y_pred, zero_division=0)),
            }
        )
    return output


def majority_predictions(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    output = np.empty_like(labels)
    for group in np.unique(groups):
        train = groups != group
        values, counts = np.unique(labels[train], return_counts=True)
        output[groups == group] = values[int(np.argmax(counts))]
    return output


def evaluate_models(
    rows: list[dict[str, object]], permutations: int, rng: np.random.Generator
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    feature_names = sorted(key for key in rows[0] if key.startswith("feature_"))
    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)
    labels = np.asarray([str(row["regime"]) for row in rows], dtype="U16")
    folds = np.asarray([int(row["broad_fold"]) for row in rows], dtype=np.int64)
    sites = np.asarray([int(row["site"]) for row in rows], dtype=np.int64)
    model_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    observed: dict[tuple[str, str], dict[str, float]] = {}

    majority = majority_predictions(labels, folds)
    majority_metrics = evaluate_predictions(labels, majority, np.full(len(labels), np.nan), False)
    for model_name in MODEL_NAMES:
        prediction, probability, local_folds = cross_validated_predictions(
            x, labels, folds, model_name, False, SEED
        )
        metrics = evaluate_predictions(labels, prediction, probability, False)
        observed[(model_name, "multiclass")] = metrics
        model_rows.append(
            {
                "model": model_name,
                "task": "multiclass",
                **metrics,
                "majority_macro_f1": majority_metrics["macro_f1"],
                "macro_f1_gain_vs_majority": metrics["macro_f1"] - majority_metrics["macro_f1"],
            }
        )
        fold_rows.extend(local_folds)
        for index, row in enumerate(rows):
            prediction_rows.append(
                {
                    "hole_id": row["hole_id"],
                    "site": row["site"],
                    "broad_fold": row["broad_fold"],
                    "task": "multiclass",
                    "model": model_name,
                    "truth": labels[index],
                    "prediction": prediction[index],
                    "probability_suppress": float("nan"),
                }
            )

    core_mask = np.isin(labels, ["no_action", "suppress"])
    core_count = int(np.sum(core_mask))
    core_counts = {
        label: int(np.sum(labels[core_mask] == label)) for label in ("no_action", "suppress")
    }
    if min(core_counts.values()) >= 2:
        xb = x[core_mask]
        yb = labels[core_mask]
        fb = folds[core_mask]
        sb = sites[core_mask]
        majority_binary = majority_predictions(yb, fb)
        majority_binary_metrics = evaluate_predictions(
            yb, majority_binary, (majority_binary == "suppress").astype(float), True
        )
        for model_name in MODEL_NAMES:
            prediction, probability, local_folds = cross_validated_predictions(
                xb, yb, fb, model_name, True, SEED
            )
            metrics = evaluate_predictions(yb, prediction, probability, True)
            observed[(model_name, "binary")] = metrics
            model_rows.append(
                {
                    "model": model_name,
                    "task": "no_action_vs_suppress",
                    **metrics,
                    "majority_balanced_accuracy": majority_binary_metrics["balanced_accuracy"],
                    "positive_fold_count": int(sum(bool(row.get("direction_positive", False)) for row in local_folds)),
                    "evaluable_fold_count": int(sum(math.isfinite(float(row.get("auc", float("nan")))) for row in local_folds)),
                }
            )
            fold_rows.extend(local_folds)
            original_indices = np.flatnonzero(core_mask)
            for local, original in enumerate(original_indices):
                row = rows[original]
                prediction_rows.append(
                    {
                        "hole_id": row["hole_id"],
                        "site": row["site"],
                        "broad_fold": row["broad_fold"],
                        "task": "no_action_vs_suppress",
                        "model": model_name,
                        "truth": yb[local],
                        "prediction": prediction[local],
                        "probability_suppress": probability[local],
                    }
                )

        permutation_values: dict[tuple[str, str, str], list[float]] = {}
        for repeat in range(permutations):
            multi_perm = labels.copy()
            for site in np.unique(sites):
                selected = np.flatnonzero(sites == site)
                multi_perm[selected] = multi_perm[rng.permutation(selected)]
            binary_perm = yb.copy()
            for site in np.unique(sb):
                selected = np.flatnonzero(sb == site)
                binary_perm[selected] = binary_perm[rng.permutation(selected)]
            for model_name in MODEL_NAMES:
                pred, prob, _ = cross_validated_predictions(
                    x, multi_perm, folds, model_name, False, SEED
                )
                metric = evaluate_predictions(multi_perm, pred, prob, False)
                permutation_values.setdefault((model_name, "multiclass", "macro_f1"), []).append(metric["macro_f1"])
                pred, prob, _ = cross_validated_predictions(
                    xb, binary_perm, fb, model_name, True, SEED
                )
                metric = evaluate_predictions(binary_perm, pred, prob, True)
                for key in ("balanced_accuracy", "auc"):
                    permutation_values.setdefault((model_name, "binary", key), []).append(metric[key])
            if (repeat + 1) % 50 == 0 or repeat + 1 == permutations:
                print(f"E172 site-stratified permutations: {repeat + 1}/{permutations}", flush=True)
        for (model_name, task, metric_name), values in permutation_values.items():
            value = np.asarray(values, dtype=np.float64)
            finite = value[np.isfinite(value)]
            observed_value = observed[(model_name, task)][metric_name]
            permutation_rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "metric": metric_name,
                    "observed": observed_value,
                    "permutation_mean": float(np.mean(finite)),
                    "permutation_p95": float(np.quantile(finite, 0.95)),
                    "empirical_p_value": float((1 + np.sum(finite >= observed_value)) / (1 + len(finite))),
                    "permutation_count": len(finite),
                }
            )

    optimistic_rows: list[dict[str, object]] = []
    hole_groups = np.arange(len(rows), dtype=np.int64)
    for model_name in MODEL_NAMES:
        prediction, probability, _ = cross_validated_predictions(
            x, labels, hole_groups, model_name, False, SEED
        )
        optimistic_rows.append(
            {
                "model": model_name,
                "task": "multiclass_leave_one_hole_out",
                **evaluate_predictions(labels, prediction, probability, False),
            }
        )
    if min(core_counts.values()) >= 2:
        xb = x[core_mask]
        yb = labels[core_mask]
        hole_groups = np.arange(len(yb), dtype=np.int64)
        for model_name in MODEL_NAMES:
            prediction, probability, _ = cross_validated_predictions(
                xb, yb, hole_groups, model_name, True, SEED
            )
            optimistic_rows.append(
                {
                    "model": model_name,
                    "task": "binary_leave_one_hole_out",
                    **evaluate_predictions(yb, prediction, probability, True),
                }
            )
    return model_rows + optimistic_rows, fold_rows, prediction_rows, permutation_rows, {
        "feature_names": feature_names,
        "core_count": core_count,
        "core_counts": core_counts,
        "multiclass_majority": majority_metrics,
    }


def feature_effects(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    no_action = [row for row in rows if row["regime"] == "no_action"]
    suppress = [row for row in rows if row["regime"] == "suppress"]
    output: list[dict[str, object]] = []
    if not no_action or not suppress:
        return output
    for name in sorted(key for key in rows[0] if key.startswith("feature_")):
        left = np.asarray([float(row[name]) for row in no_action], dtype=np.float64)
        right = np.asarray([float(row[name]) for row in suppress], dtype=np.float64)
        combined = np.concatenate([left, right])
        finite = np.isfinite(combined)
        if np.sum(finite) < len(combined):
            median = float(np.nanmedian(combined))
            left = np.where(np.isfinite(left), left, median)
            right = np.where(np.isfinite(right), right, median)
        pooled = math.sqrt(max((np.var(left) + np.var(right)) / 2.0, 1e-12))
        effect = float((np.mean(right) - np.mean(left)) / pooled)
        truth = np.concatenate([np.zeros(len(left)), np.ones(len(right))])
        auc = float(roc_auc_score(truth, np.concatenate([left, right])))
        output.append(
            {
                "feature": name,
                "no_action_mean": float(np.mean(left)),
                "suppress_mean": float(np.mean(right)),
                "cohen_d_suppress_minus_no_action": effect,
                "oriented_univariate_auc": max(auc, 1.0 - auc),
                "higher_value_class": "suppress" if auc >= 0.5 else "no_action",
            }
        )
    return sorted(output, key=lambda row: float(row["oriented_univariate_auc"]), reverse=True)


def cluster_regimes(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    shares = np.asarray(
        [[float(row[f"{name}_share"]) for name in ERROR_TYPES] for row in rows],
        dtype=np.float64,
    )
    trials = []
    maximum = min(6, len(rows) - 1)
    for clusters in range(2, maximum + 1):
        labels = KMeans(n_clusters=clusters, n_init=50, random_state=SEED).fit_predict(shares)
        trials.append({"clusters": clusters, "silhouette": float(silhouette_score(shares, labels))})
    best = max(trials, key=lambda row: float(row["silhouette"]))
    count = int(best["clusters"])
    kmeans = KMeans(n_clusters=count, n_init=50, random_state=SEED).fit_predict(shares)
    hierarchical = AgglomerativeClustering(n_clusters=count, linkage="ward").fit_predict(shares)
    regime_ids = np.asarray([REGIME_ORDER.index(str(row["regime"])) for row in rows])
    output = []
    for index, row in enumerate(rows):
        output.append(
            {
                "hole_id": row["hole_id"],
                "regime": row["regime"],
                "kmeans_cluster": int(kmeans[index]),
                "hierarchical_cluster": int(hierarchical[index]),
                **{f"{name}_share": shares[index, position] for position, name in enumerate(ERROR_TYPES)},
            }
        )
    return output, {
        "silhouette_trials": trials,
        "best_cluster_count": count,
        "best_silhouette": float(best["silhouette"]),
        "kmeans_hierarchical_ari": float(adjusted_rand_score(kmeans, hierarchical)),
        "kmeans_regime_ari": float(adjusted_rand_score(kmeans, regime_ids)),
    }


def make_plots(
    rows: list[dict[str, object]],
    feature_effect_rows: list[dict[str, object]],
    cluster_summary: dict[str, object],
    prediction_rows: list[dict[str, object]],
) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (REGIME_ORDER.index(str(row["regime"])), str(row["hole_id"])))
    shares = np.asarray(
        [[float(row[f"{name}_share"]) for name in ERROR_TYPES] for row in ordered], dtype=np.float64
    )
    fig, ax = plt.subplots(figsize=(9, 10))
    image = ax.imshow(shares, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.6, float(np.max(shares))))
    ax.set_xticks(range(4), ["missing", "spurious", "shift", "power"])
    ax.set_yticks(range(len(ordered)), [f"{row['hole_id']} | {row['regime']}" for row in ordered], fontsize=7)
    ax.set_title("E172 whole-hole PAS error composition")
    fig.colorbar(image, ax=ax, label="error mass share")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "01_洞级错误质量构成.png", dpi=180)
    plt.close(fig)

    trials = cluster_summary["silhouette_trials"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["clusters"] for row in trials], [row["silhouette"] for row in trials], marker="o")
    ax.set_xlabel("cluster count")
    ax.set_ylabel("silhouette")
    ax.set_title("Error-share cluster separation")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIS_DIR / "02_洞级错误构成聚类轮廓.png", dpi=180)
    plt.close(fig)

    top = feature_effect_rows[:6]
    if top:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, item in zip(axes.ravel(), top):
            name = str(item["feature"])
            left = [float(row[name]) for row in rows if row["regime"] == "no_action"]
            right = [float(row[name]) for row in rows if row["regime"] == "suppress"]
            ax.boxplot([left, right], tick_labels=["no_action", "suppress"], showfliers=False)
            ax.set_title(name.replace("feature_", ""), fontsize=9)
            ax.tick_params(axis="x", labelrotation=15)
        fig.suptitle("Top Clean feature differences")
        fig.tight_layout()
        fig.savefig(VIS_DIR / "03_压峰洞与不动洞Clean特征分布.png", dpi=180)
        plt.close(fig)

    binary = [
        row
        for row in prediction_rows
        if row["task"] == "no_action_vs_suppress" and row["model"] == "extra_trees"
    ]
    if binary:
        truth = [row["truth"] for row in binary]
        prediction = [row["prediction"] for row in binary]
        matrix = confusion_matrix(truth, prediction, labels=["no_action", "suppress"])
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["no_action", "suppress"])
        ax.set_yticks([0, 1], ["no_action", "suppress"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("truth")
        ax.set_title("Strict leave-broad-fold-out")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(VIS_DIR / "04_严格整外折压峰与不动混淆矩阵.png", dpi=180)
        plt.close(fig)


def decision_summary(
    rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    permutation_rows: list[dict[str, object]],
) -> dict[str, object]:
    stability = np.asarray([float(row["bootstrap_stability"]) for row in rows])
    counts = {name: int(sum(row["regime"] == name for row in rows)) for name in REGIME_ORDER}
    binary_models = [row for row in model_rows if row["task"] == "no_action_vs_suppress"]
    multiclass_models = [row for row in model_rows if row["task"] == "multiclass"]
    best_binary = max(binary_models, key=lambda row: float(row.get("balanced_accuracy", -1.0))) if binary_models else None
    best_multi = max(multiclass_models, key=lambda row: float(row.get("macro_f1", -1.0))) if multiclass_models else None
    permutation_lookup = {
        (row["model"], row["task"], row["metric"]): row for row in permutation_rows
    }
    existence_pass = (
        len(rows) >= 30
        and float(np.median(stability)) >= 0.70
        and counts["no_action"] >= 5
        and counts["suppress"] >= 5
    )
    predictability_pass = False
    if best_binary is not None:
        control_ba = permutation_lookup.get((best_binary["model"], "binary", "balanced_accuracy"), {})
        control_auc = permutation_lookup.get((best_binary["model"], "binary", "auc"), {})
        predictability_pass = bool(
            float(best_binary["balanced_accuracy"]) >= 0.70
            and float(best_binary["auc"]) >= 0.75
            and float(best_binary["balanced_accuracy"]) > float(control_ba.get("permutation_p95", np.inf))
            and float(best_binary["auc"]) > float(control_auc.get("permutation_p95", np.inf))
            and int(best_binary.get("positive_fold_count", 0)) >= 4
        )
    multiclass_pass = bool(
        best_multi is not None and float(best_multi["macro_f1_gain_vs_majority"]) >= 0.15
    )
    return {
        "hole_count": len(rows),
        "regime_counts": counts,
        "median_bootstrap_stability": float(np.median(stability)),
        "regimes_exist_gate": existence_pass,
        "best_binary": best_binary,
        "binary_predictable_gate": predictability_pass,
        "best_multiclass": best_multi,
        "multiclass_useful_gate": multiclass_pass,
    }


def write_report(report: dict[str, object]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(json_compatible(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gate = report["proxy_gate"]
    if not gate["passed"]:
        text = f"""# E172：洞级修正regime稳定性审计

## 先说结论

实验在第一道代理门停止。用于扩展洞的严格 `V7-core` 与完整V7在原10个标准洞上的regime一致率为 `{gate['regime_agreement']:.3f}`，四类错误占比向量平均余弦为 `{gate['mean_error_share_cosine']:.3f}`；预注册要求分别为 `0.70/0.85`。

这意味着扩展洞上的标签不能可靠代表完整V7的洞级修法。继续做45洞分类只会回答“严格整折V7-core是什么规律”，不能回答用户提出的“完整V7的洞级规律稳不稳”，因此按合同停止。

## 官方分差异

| 表示 | PAS | PDP | NMSE | 总分 |
|---|---:|---:|---:|---:|
| 完整V7 fixed500 | {gate['complete_overall']['pas']:.6f} | {gate['complete_overall']['pdp']:.6f} | {gate['complete_overall']['nmse']:.6f} | {gate['complete_overall']['score']:.6f} |
| 严格V7-core 同500点 | {gate['core_same500_overall']['pas']:.6f} | {gate['core_same500_overall']['pdp']:.6f} | {gate['core_same500_overall']['nmse']:.6f} | {gate['core_same500_overall']['score']:.6f} |

本轮未训练模型、未使用Sionna、未生成候选版本。
"""
        REPORT_MD.write_text(text, encoding="utf-8")
        return

    decision = report["decision"]
    cluster = report["cluster_summary"]
    best_binary = decision.get("best_binary")
    best_multi = decision.get("best_multiclass")
    binary_line = (
        f"最佳严格二分类是 `{best_binary['model']}`：BA=`{best_binary['balanced_accuracy']:.3f}`，"
        f"AUC=`{best_binary['auc']:.3f}`，precision=`{best_binary['precision_suppress']:.3f}`，"
        f"recall=`{best_binary['recall_suppress']:.3f}`。"
        if best_binary
        else "`no_action/suppress`样本不足，未形成严格二分类结果。"
    )
    multi_line = (
        f"最佳严格多分类是 `{best_multi['model']}`：macro-F1=`{best_multi['macro_f1']:.3f}`，"
        f"比整折多数类基线高`{best_multi['macro_f1_gain_vs_majority']:+.3f}`。"
        if best_multi
        else "未形成多分类结果。"
    )
    text = f"""# E172：峰级规律跨洞失效后，洞级规律是否稳定

## 最关键结论

共得到 `{decision['hole_count']}` 个严格Clean pseudo-hole，洞级标签bootstrap稳定率中位数为 `{decision['median_bootstrap_stability']:.3f}`。标签数量为：`{decision['regime_counts']}`。

1. **不同洞是否确实需要不同修法：** {'是，错误构成和稳定标签通过预注册存在性门。' if decision['regimes_exist_gate'] else '证据不足，样本数量、类别数量或标签稳定性没有同时过门。'}
2. **Clean信息能否在未见区域判断修法：** {'通过严格整外折门。' if decision['binary_predictable_gate'] else '没有通过严格整外折可预测性门。'} {binary_line}
3. **多类修法能否直接判断：** {'达到预注册可用门槛。' if decision['multiclass_useful_gate'] else '没有达到预注册可用门槛。'} {multi_line}

错误占比本身的最佳KMeans轮廓系数为 `{cluster['best_silhouette']:.3f}`（`k={cluster['best_cluster_count']}`），KMeans与层次聚类ARI为 `{cluster['kmeans_hierarchical_ari']:.3f}`。这只说明错误构成有没有自然分组，不替代严格未见折分类。

## 代理门

完整V7与严格V7-core在原10洞上的regime一致率=`{gate['regime_agreement']:.3f}`，四类错误占比余弦=`{gate['mean_error_share_cosine']:.3f}`，代理门通过。

| 表示 | PAS | PDP | NMSE | 总分 |
|---|---:|---:|---:|---:|
| 完整V7 fixed500 | {gate['complete_overall']['pas']:.6f} | {gate['complete_overall']['pdp']:.6f} | {gate['complete_overall']['nmse']:.6f} | {gate['complete_overall']['score']:.6f} |
| 严格V7-core 同500点 | {gate['core_same500_overall']['pas']:.6f} | {gate['core_same500_overall']['pdp']:.6f} | {gate['core_same500_overall']['nmse']:.6f} | {gate['core_same500_overall']['score']:.6f} |

## 边界

扩展洞点数范围只有 `{report['hole_construction']['point_count_min']}-{report['hole_construction']['point_count_max']}`，不能充分覆盖真实Test中`50-90`点的大洞。本轮结论只回答洞级规律在现有Clean可构造洞上是否稳定，不把代理分类指标当作线上分数，也没有修改任何H。

本轮未使用Sionna、V8-V12或其派生信息，未训练大模型，未生成提交版本。
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    start_time = time.time()
    rng = np.random.default_rng(SEED)
    device = torch.device(args.device)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fields = build_full_fields(device, args.batch_size, args.force_fields)
    proxy_gate, proxy_rows = canonical_proxy_gate(fields, rng)
    write_csv(PROXY_CSV, proxy_rows)
    report: dict[str, object] = {
        "version": VERSION,
        "clean_lineage": {
            "allowed": [
                str(TRAIN_CHANNEL.relative_to(PATHS.root)),
                str((PATHS.round2 / "Round2_Train_Pos.npy").relative_to(PATHS.root)),
                str((PATHS.round2 / "Round2_Test_Pos.npy").relative_to(PATHS.root)),
                str((PATHS.round2 / "Round2_Map.ply").relative_to(PATHS.root)),
                str((PATHS.round2 / "Round2_Setup.json").relative_to(PATHS.root)),
                str(FULL_CORE.relative_to(PATHS.root)),
            ],
            "sionna_used": False,
            "v8_v12_used": False,
            "test_h_used": False,
        },
        "proxy_gate": proxy_gate,
    }
    if not bool(proxy_gate["passed"]):
        report["status"] = "stopped_at_proxy_gate"
        report["elapsed_seconds"] = time.time() - start_time
        write_report(report)
        print(json.dumps(json_compatible(report), ensure_ascii=False, indent=2), flush=True)
        return

    holes, construction = build_expanded_holes()
    report["hole_construction"] = construction
    if len(holes) < 30:
        report["status"] = "stopped_at_hole_count_gate"
        report["elapsed_seconds"] = time.time() - start_time
        write_report(report)
        print(json.dumps(json_compatible(report), ensure_ascii=False, indent=2), flush=True)
        return

    regime_rows, _ = profile_expanded_holes(holes, fields, rng)
    feature_rows = extract_hole_features(holes, fields)
    real_holes, _, context = e167.build_holes()
    combined_rows = attach_rows(holes, regime_rows, feature_rows, context)
    match_rows = real_hole_matches(real_holes, combined_rows, context)
    cluster_rows, cluster_summary = cluster_regimes(combined_rows)
    model_rows, fold_rows, prediction_rows, permutation_rows, model_context = evaluate_models(
        combined_rows, args.permutations, rng
    )
    effect_rows = feature_effects(combined_rows)
    decision = decision_summary(combined_rows, model_rows, permutation_rows)

    write_csv(HOLE_CSV, combined_rows)
    write_csv(FEATURE_CSV, feature_rows)
    write_csv(MATCH_CSV, match_rows)
    write_csv(CLUSTER_CSV, cluster_rows)
    write_csv(MODEL_CSV, model_rows)
    write_csv(FOLD_CSV, fold_rows)
    write_csv(PREDICTION_CSV, prediction_rows)
    write_csv(PERMUTATION_CSV, permutation_rows)
    write_csv(FEATURE_EFFECT_CSV, effect_rows)
    make_plots(combined_rows, effect_rows, cluster_summary, prediction_rows)

    report.update(
        {
            "status": "analysis_complete",
            "hole_construction": construction,
            "cluster_summary": cluster_summary,
            "model_context": model_context,
            "model_results": model_rows,
            "permutation_control": permutation_rows,
            "decision": decision,
            "top_feature_effects": effect_rows[:12],
            "output_files": [
                str(path.relative_to(PATHS.root))
                for path in (
                    REPORT_JSON,
                    REPORT_MD,
                    PROXY_CSV,
                    HOLE_CSV,
                    FEATURE_CSV,
                    MATCH_CSV,
                    MODEL_CSV,
                    FOLD_CSV,
                    PREDICTION_CSV,
                    PERMUTATION_CSV,
                    FEATURE_EFFECT_CSV,
                    CLUSTER_CSV,
                )
            ],
            "elapsed_seconds": time.time() - start_time,
        }
    )
    write_report(report)
    print(json.dumps(json_compatible(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
