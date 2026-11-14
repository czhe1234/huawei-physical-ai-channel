#!/usr/bin/env python3
"""Test whether Clean observables can identify V7 PAS peaks safe to suppress."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

import analyze_e167_clean_test_hole_difficulty as e167
import analyze_e169_hard_hole_pas_peak_precursors as e169
from common.official_metrics import PointMetricStatistics, aggregate_statistics, point_metric_statistics
from project_paths import PATHS


VERSION = "Round2-E171-v7-clean-peak-suppress"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e171_v7_clean_peak_suppress"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e171_v7_clean_peak_suppress"
VIS_DIR = PATHS.analysis / "visualizations" / "复赛V7假峰Clean高精度压峰审计"

FEATURE_CACHE = ARTIFACT_DIR / "v7_clean_peak_features.npz"
REPORT_JSON = ANALYSIS_DIR / "v7_clean_peak_suppress.json"
REPORT_MD = ANALYSIS_DIR / "V7假峰能否靠Clean信息露馅.md"
LABEL_CSV = ANALYSIS_DIR / "V7峰标签统计.csv"
GROUP_CSV = ANALYSIS_DIR / "特征组与模型OOF效果.csv"
HOLE_CLASSIFICATION_CSV = ANALYSIS_DIR / "逐洞分类效果.csv"
FEATURE_CSV = ANALYSIS_DIR / "单特征Clean信息量.csv"
NESTED_CSV = ANALYSIS_DIR / "嵌套90precision逐洞选择.csv"
SUPPRESSION_CSV = ANALYSIS_DIR / "保守压峰fixed500完整分.csv"
SUPPRESSION_HOLE_CSV = ANALYSIS_DIR / "保守压峰逐洞官方分.csv"
PEAK_PREDICTION_CSV = ANALYSIS_DIR / "V7逐峰OOF预测.csv"

V7_CHANNEL = e167.V7_COMPLETE
TRAIN_CHANNEL = e167.TRAIN_CHANNEL
OOF_META = e167.OOF_META
TRAIN_COMPACT = e169.TRAIN_COMPACT
V7_COMPACT = e169.V7_COMPACT
CARRIER_BANK = PATHS.artifacts / "round2" / "complex_carrier_bank" / "oof_complex_carrier_bank.npy"
CARRIER_META = PATHS.artifacts / "round2" / "complex_carrier_bank" / "complex_carrier_bank_metadata.json"
MAP_FEATURES = PATHS.artifacts / "round2" / "regime_lifecycle_support" / "geodesic_corridor_features.npz"
SPECTRAL_DESCRIPTORS = PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_spectral_descriptors.npz"
POINT_FEATURES = e167.POINT_CSV

KEEP = 0
SUPPRESS = 1
AMBIGUOUS = -1
TARGET_PRECISIONS = (0.80, 0.90, 0.95)
SUPPRESSION_LEVELS = (0.10, 0.20, 0.30, 0.50)
NEIGHBORS = 32
NEAR_COUNT = 8
FAR_START = 16
TINY = 1e-30
SEED = 20_260_811_171

# Frozen before fitting. GT is used only to create labels inside pseudo-holes.
KEEP_MIN_MASS = 0.015
KEEP_RELATIVE_MASS = 0.40
SUPPRESS_MAX_MASS = 0.012
SUPPRESS_RELATIVE_MASS = 0.25

FEATURE_GROUPS = {
    "self": ("self_",),
    "carrier": ("carrier_",),
    "neighbor": ("neighbor_",),
    "cross_domain": ("context_",),
    "geometry": ("geo_",),
    "carrier_neighbor": ("self_", "carrier_", "neighbor_"),
    "all": ("self_", "carrier_", "neighbor_", "context_", "geo_"),
}
MODELS = ("logistic", "extra_trees", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("features", "audit", "all"), default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-features", action="store_true")
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


def periodic_distance(left: np.ndarray | float, right: np.ndarray | float, period: int) -> np.ndarray:
    delta = np.abs(np.asarray(left) - np.asarray(right))
    return np.minimum(delta, period - delta)


def circular_concentration(indices: np.ndarray, period: int, weights: np.ndarray | None = None) -> float:
    values = np.asarray(indices, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    angle = 2.0 * np.pi * values / period
    unit = np.exp(1j * angle)
    if weights is None:
        return float(abs(np.mean(unit)))
    weight = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if float(np.sum(weight)) <= 0.0:
        return float(abs(np.mean(unit)))
    return float(abs(np.sum(weight * unit) / np.sum(weight)))


def entropy(value: np.ndarray) -> float:
    probability = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = float(np.sum(probability))
    if total <= 0.0:
        return 0.0
    probability /= total
    valid = probability > 0.0
    return float(-np.sum(probability[valid] * np.log(probability[valid])) / math.log(len(probability)))


def add_feature(columns: dict[str, np.ndarray], name: str, value: np.ndarray | list[float]) -> None:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"feature {name} must be one-dimensional, got {array.shape}")
    columns[name] = array


def point_context() -> dict[str, object]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, pseudo_holes, context = e167.build_holes()
    train_positions = np.asarray(context["train_positions"], dtype=np.float64)
    global_to_oof = np.full(len(train_positions), -1, dtype=np.int64)
    global_to_oof[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    hole_id = np.empty(500, dtype="U16")
    site = np.empty(500, dtype=np.int8)
    neighbor_ids = np.empty((500, NEIGHBORS), dtype=np.int64)
    neighbor_distances = np.empty((500, NEIGHBORS), dtype=np.float32)
    for hole in pseudo_holes:
        rows = global_to_oof[hole.point_ids]
        if np.any(rows < 0) or np.any(fold_id[rows] != hole.ordinal):
            raise RuntimeError(f"canonical hole mismatch: {hole.hole_id}")
        hole_id[rows] = hole.hole_id
        site[rows] = hole.site
        tree = cKDTree(train_positions[hole.fit_ids, :2])
        distance, local = tree.query(train_positions[hole.point_ids, :2], k=NEIGHBORS)
        neighbor_ids[rows] = hole.fit_ids[local]
        neighbor_distances[rows] = distance
    point_rows: list[dict[str, str] | None] = [None] * 500
    with POINT_FEATURES.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            point_rows[int(row["oof_row"])] = row
    if any(row is None for row in point_rows):
        raise RuntimeError("E167 point features do not cover all OOF points")
    return {
        "query_ids": query_ids,
        "fold_id": fold_id,
        "hole_id": hole_id,
        "site": site,
        "neighbor_ids": neighbor_ids,
        "neighbor_distances": neighbor_distances,
        "point_rows": point_rows,
        "train_positions": train_positions,
        "context": context,
    }


def build_peak_samples(v7: np.ndarray, truth: np.ndarray, context: dict[str, object]) -> dict[str, np.ndarray]:
    rows: list[dict[str, object]] = []
    hole_id = np.asarray(context["hole_id"])
    site = np.asarray(context["site"])
    query_ids = np.asarray(context["query_ids"])
    for point in range(500):
        for ue in range(e169.N_UE):
            pred_peaks = e169.extract_peaks(v7[point, ue])
            truth_peaks = e169.extract_peaks(truth[point, ue])
            for rank, peak in enumerate(pred_peaks):
                gt_mass = e169.window_mass(truth[point, ue], peak.h, peak.v)
                nearest = min(
                    (e169.scaled_distance(peak, target)[0] for target in truth_peaks),
                    default=float("inf"),
                )
                if nearest <= 1.0 and gt_mass >= max(KEEP_MIN_MASS, KEEP_RELATIVE_MASS * peak.mass):
                    label = KEEP
                elif nearest > 1.0 and gt_mass <= min(SUPPRESS_MAX_MASS, SUPPRESS_RELATIVE_MASS * peak.mass):
                    label = SUPPRESS
                else:
                    label = AMBIGUOUS
                rows.append(
                    {
                        "sample_id": len(rows),
                        "oof_row": point,
                        "global_train_id": int(query_ids[point]),
                        "hole_id": str(hole_id[point]),
                        "site": int(site[point]),
                        "ue": ue,
                        "h": peak.h,
                        "v": peak.v,
                        "rank": rank,
                        "v7_peak_value": peak.value,
                        "v7_mass": peak.mass,
                        "gt_mass": gt_mass,
                        "nearest_gt_peak_distance": nearest,
                        "label": label,
                        "suppress_error_mass": max(peak.mass - gt_mass, 0.0) if label == SUPPRESS else 0.0,
                    }
                )
    result: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key == "hole_id":
            result[key] = np.asarray(values, dtype="U16")
        elif key in {"sample_id", "oof_row", "global_train_id", "site", "ue", "h", "v", "rank", "label"}:
            result[key] = np.asarray(values, dtype=np.int64)
        else:
            result[key] = np.asarray(values, dtype=np.float32)
    return result


def sample_ids_by_point(samples: dict[str, np.ndarray]) -> list[np.ndarray]:
    point = samples["oof_row"]
    return [np.flatnonzero(point == index).astype(np.int64) for index in range(500)]


def self_features(samples: dict[str, np.ndarray], v7: np.ndarray) -> dict[str, np.ndarray]:
    count = len(samples["sample_id"])
    columns: dict[str, np.ndarray] = {}
    peak_count = np.empty(count, dtype=np.float32)
    map_entropy = np.empty(count, dtype=np.float32)
    top1 = np.empty(count, dtype=np.float32)
    top4 = np.empty(count, dtype=np.float32)
    outer_mass = np.empty(count, dtype=np.float32)
    contrast = np.empty(count, dtype=np.float32)
    for index in range(count):
        point = int(samples["oof_row"][index])
        ue = int(samples["ue"][index])
        h = int(samples["h"][index])
        v = int(samples["v"][index])
        value = v7[point, ue]
        flat = np.sort(value.reshape(-1))[::-1]
        peaks = e169.extract_peaks(value)
        h5 = e169.angle_indices(h, 2, e169.N_H)
        v5 = e169.angle_indices(v, 2, e169.N_V)
        total5 = float(value[np.ix_(h5, v5)].sum())
        ring = max(total5 - float(samples["v7_mass"][index]), 0.0)
        peak_count[index] = len(peaks)
        map_entropy[index] = entropy(value.reshape(-1))
        top1[index] = float(flat[0])
        top4[index] = float(np.sum(flat[:4]))
        outer_mass[index] = ring
        contrast[index] = float(samples["v7_mass"][index]) / max(ring, 1e-6)
    add_feature(columns, "self_peak_value", samples["v7_peak_value"])
    add_feature(columns, "self_window_mass", samples["v7_mass"])
    add_feature(columns, "self_rank_normalized", samples["rank"] / max(e169.MAX_PEAKS - 1, 1))
    add_feature(columns, "self_peak_count", peak_count)
    add_feature(columns, "self_map_entropy", map_entropy)
    add_feature(columns, "self_point_top1", top1)
    add_feature(columns, "self_point_top4", top4)
    add_feature(columns, "self_outer_ring_mass", outer_mass)
    add_feature(columns, "self_local_contrast", contrast)
    add_feature(columns, "self_h_sin", np.sin(2.0 * np.pi * samples["h"] / e169.N_H))
    add_feature(columns, "self_h_cos", np.cos(2.0 * np.pi * samples["h"] / e169.N_H))
    add_feature(columns, "self_v_sin", np.sin(2.0 * np.pi * samples["v"] / e169.N_V))
    add_feature(columns, "self_v_cos", np.cos(2.0 * np.pi * samples["v"] / e169.N_V))
    return columns


def compute_v7_reference(
    samples: dict[str, np.ndarray],
    channel: np.ndarray,
    by_point: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    count = len(samples["sample_id"])
    reference = np.empty((count, 2, 192), dtype=np.complex64)
    band_mass = np.empty((count, 4), dtype=np.float32)
    delay_bin = np.empty(count, dtype=np.float32)
    delay_top1 = np.empty(count, dtype=np.float32)
    delay_top4 = np.empty(count, dtype=np.float32)
    delay_entropy = np.empty(count, dtype=np.float32)
    frequency_coherence = np.empty(count, dtype=np.float32)
    for start in range(0, 500, batch_size):
        stop = min(start + batch_size, 500)
        sample_ids = np.concatenate(by_point[start:stop]) if any(len(x) for x in by_point[start:stop]) else np.empty(0, dtype=np.int64)
        value = torch.as_tensor(
            np.array(channel[start:stop], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            shaped = value.reshape(len(value), 2, 16, 8, 4, 192)
            angle = torch.fft.fft2(shaped, dim=(2, 3))
            power = angle.abs().square().float()
            band_maps = []
            for band in range(4):
                left = 48 * band
                right = 48 * (band + 1)
                local = power[..., left:right].mean(dim=-1)
                local = local / local.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
                band_maps.append(local.sum(dim=1).permute(0, 3, 1, 2).cpu().numpy())
            if len(sample_ids):
                local_point = torch.as_tensor(samples["oof_row"][sample_ids] - start, device=device)
                h = torch.as_tensor(samples["h"][sample_ids], device=device)
                v = torch.as_tensor(samples["v"][sample_ids], device=device)
                ue = torch.as_tensor(samples["ue"][sample_ids], device=device)
                vectors = angle[local_point, :, h, v, ue, :]
                cir = torch.fft.ifft(vectors, dim=-1)
                delay = cir.abs().square().sum(dim=1).float()
                delay = delay / delay.sum(dim=1, keepdim=True).clamp_min(TINY)
                adjacent = (vectors[:, :, 1:] * vectors[:, :, :-1].conj()).sum(dim=(1, 2))
                adjacent_den = torch.sqrt(
                    vectors[:, :, 1:].abs().square().sum(dim=(1, 2))
                    * vectors[:, :, :-1].abs().square().sum(dim=(1, 2))
                ).clamp_min(TINY)
                reference[sample_ids] = vectors.cpu().numpy()
                delay_cpu = delay.cpu().numpy()
                delay_bin[sample_ids] = np.argmax(delay_cpu, axis=1)
                delay_top1[sample_ids] = np.max(delay_cpu, axis=1)
                delay_top4[sample_ids] = np.partition(delay_cpu, -4, axis=1)[:, -4:].sum(axis=1)
                delay_entropy[sample_ids] = np.asarray([entropy(row) for row in delay_cpu])
                frequency_coherence[sample_ids] = (adjacent.abs() / adjacent_den).cpu().numpy()
        for sample_id in sample_ids:
            point = int(samples["oof_row"][sample_id]) - start
            ue = int(samples["ue"][sample_id])
            h = int(samples["h"][sample_id])
            v = int(samples["v"][sample_id])
            hi = e169.angle_indices(h, 1, e169.N_H)
            vi = e169.angle_indices(v, 1, e169.N_V)
            for band in range(4):
                band_mass[sample_id, band] = band_maps[band][point, ue][np.ix_(hi, vi)].sum()
        if stop % 100 == 0 or stop == 500:
            print(f"E171 V7 cross-domain features: {stop}/500", flush=True)
    columns: dict[str, np.ndarray] = {}
    for band in range(4):
        add_feature(columns, f"context_frequency_band{band}_mass", band_mass[:, band])
    add_feature(columns, "context_frequency_mass_mean", np.mean(band_mass, axis=1))
    add_feature(columns, "context_frequency_mass_std", np.std(band_mass, axis=1))
    add_feature(columns, "context_frequency_mass_min", np.min(band_mass, axis=1))
    add_feature(columns, "context_frequency_mass_range", np.ptp(band_mass, axis=1))
    add_feature(columns, "context_v7_delay_bin_sin", np.sin(2.0 * np.pi * delay_bin / 192.0))
    add_feature(columns, "context_v7_delay_bin_cos", np.cos(2.0 * np.pi * delay_bin / 192.0))
    add_feature(columns, "context_v7_delay_top1", delay_top1)
    add_feature(columns, "context_v7_delay_top4", delay_top4)
    add_feature(columns, "context_v7_delay_entropy", delay_entropy)
    add_feature(columns, "context_v7_subcarrier_coherence", frequency_coherence)
    return reference, columns


def compute_carrier_features(
    samples: dict[str, np.ndarray],
    reference: np.ndarray,
    by_point: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    metadata = json.loads(CARRIER_META.read_text(encoding="utf-8"))
    if not bool(metadata.get("official_inputs_only")) or bool(metadata.get("sionna_used")):
        raise RuntimeError("carrier bank lineage is not Clean")
    bank = np.load(CARRIER_BANK, mmap_mode="r")
    if bank.shape != (17, 500, 256, 4, 192) or bank.dtype != np.complex64:
        raise ValueError(f"unexpected carrier bank: {bank.shape}/{bank.dtype}")
    count = len(samples["sample_id"])
    carrier_mass = np.empty((count, 17), dtype=np.float32)
    carrier_coherence = np.empty((count, 17), dtype=np.float32)
    carrier_phase = np.empty((count, 17), dtype=np.float32)
    carrier_delay_bin = np.empty((count, 17), dtype=np.float32)
    carrier_delay_top1 = np.empty((count, 17), dtype=np.float32)
    carrier_delay_entropy = np.empty((count, 17), dtype=np.float32)
    for carrier in range(17):
        for start in range(0, 500, batch_size):
            stop = min(start + batch_size, 500)
            sample_ids = np.concatenate(by_point[start:stop]) if any(len(x) for x in by_point[start:stop]) else np.empty(0, dtype=np.int64)
            value = torch.as_tensor(
                np.array(bank[carrier, start:stop], dtype=np.complex64, copy=True), device=device
            )
            with torch.inference_mode():
                shaped = value.reshape(len(value), 2, 16, 8, 4, 192)
                angle = torch.fft.fft2(shaped, dim=(2, 3))
                power = angle.abs().square().float()
                compact = power.sum(dim=1).mean(dim=-1).permute(0, 3, 1, 2)
                compact = compact / compact.sum(dim=(2, 3), keepdim=True).clamp_min(TINY)
                compact_cpu = compact.cpu().numpy()
                if len(sample_ids):
                    local_point = torch.as_tensor(samples["oof_row"][sample_ids] - start, device=device)
                    h = torch.as_tensor(samples["h"][sample_ids], device=device)
                    v = torch.as_tensor(samples["v"][sample_ids], device=device)
                    ue = torch.as_tensor(samples["ue"][sample_ids], device=device)
                    vectors = angle[local_point, :, h, v, ue, :]
                    ref = torch.as_tensor(reference[sample_ids], device=device)
                    inner = (vectors * ref.conj()).sum(dim=(1, 2))
                    denominator = torch.sqrt(
                        vectors.abs().square().sum(dim=(1, 2))
                        * ref.abs().square().sum(dim=(1, 2))
                    ).clamp_min(TINY)
                    coherence = inner.abs() / denominator
                    delay = torch.fft.ifft(vectors, dim=-1).abs().square().sum(dim=1).float()
                    delay = delay / delay.sum(dim=1, keepdim=True).clamp_min(TINY)
                    delay_cpu = delay.cpu().numpy()
                    carrier_coherence[sample_ids, carrier] = coherence.cpu().numpy()
                    carrier_phase[sample_ids, carrier] = torch.angle(inner).cpu().numpy()
                    carrier_delay_bin[sample_ids, carrier] = np.argmax(delay_cpu, axis=1)
                    carrier_delay_top1[sample_ids, carrier] = np.max(delay_cpu, axis=1)
                    carrier_delay_entropy[sample_ids, carrier] = np.asarray([entropy(row) for row in delay_cpu])
            for sample_id in sample_ids:
                point = int(samples["oof_row"][sample_id]) - start
                ue_value = int(samples["ue"][sample_id])
                h_value = int(samples["h"][sample_id])
                v_value = int(samples["v"][sample_id])
                carrier_mass[sample_id, carrier] = e169.window_mass(
                    compact_cpu[point, ue_value], h_value, v_value
                )
        print(f"E171 carrier features: {carrier + 1}/17", flush=True)
    columns: dict[str, np.ndarray] = {}
    for carrier in range(17):
        add_feature(columns, f"carrier_mass_{carrier:02d}", carrier_mass[:, carrier])
    v7_mass = samples["v7_mass"][:, None]
    add_feature(columns, "carrier_mass_mean", np.mean(carrier_mass, axis=1))
    add_feature(columns, "carrier_mass_std", np.std(carrier_mass, axis=1))
    add_feature(columns, "carrier_mass_min", np.min(carrier_mass, axis=1))
    add_feature(columns, "carrier_mass_max", np.max(carrier_mass, axis=1))
    add_feature(columns, "carrier_mass_q25", np.quantile(carrier_mass, 0.25, axis=1))
    add_feature(columns, "carrier_mass_q75", np.quantile(carrier_mass, 0.75, axis=1))
    add_feature(columns, "carrier_mass_cv", np.std(carrier_mass, axis=1) / np.maximum(np.mean(carrier_mass, axis=1), 1e-6))
    add_feature(columns, "carrier_support_abs005_fraction", np.mean(carrier_mass >= 0.005, axis=1))
    add_feature(columns, "carrier_support_abs010_fraction", np.mean(carrier_mass >= 0.010, axis=1))
    add_feature(columns, "carrier_support_half_v7_fraction", np.mean(carrier_mass >= 0.5 * v7_mass, axis=1))
    add_feature(columns, "carrier_within_factor2_v7_fraction", np.mean((carrier_mass >= 0.5 * v7_mass) & (carrier_mass <= 2.0 * v7_mass), axis=1))
    add_feature(columns, "carrier_coherence_mean", np.mean(carrier_coherence, axis=1))
    add_feature(columns, "carrier_coherence_std", np.std(carrier_coherence, axis=1))
    add_feature(columns, "carrier_coherence_min", np.min(carrier_coherence, axis=1))
    add_feature(columns, "carrier_coherence_q25", np.quantile(carrier_coherence, 0.25, axis=1))
    add_feature(columns, "carrier_high_coherence_fraction", np.mean(carrier_coherence >= 0.8, axis=1))
    add_feature(columns, "carrier_phase_concentration", np.abs(np.mean(np.exp(1j * carrier_phase), axis=1)))
    add_feature(columns, "carrier_delay_concentration", np.asarray([circular_concentration(row, 192) for row in carrier_delay_bin]))
    add_feature(columns, "carrier_delay_top1_mean", np.mean(carrier_delay_top1, axis=1))
    add_feature(columns, "carrier_delay_top1_std", np.std(carrier_delay_top1, axis=1))
    add_feature(columns, "carrier_delay_entropy_mean", np.mean(carrier_delay_entropy, axis=1))
    add_feature(columns, "carrier_delay_entropy_std", np.std(carrier_delay_entropy, axis=1))
    return columns


def cosine_dispersion(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(array, axis=1)
    valid = norm > 0.0
    if np.sum(valid) < 2:
        return 0.0
    unit = array[valid] / norm[valid, None]
    center = np.mean(unit, axis=0)
    center_norm = np.linalg.norm(center)
    if center_norm <= 0.0:
        return 1.0
    return float(np.mean(1.0 - unit @ (center / center_norm)))


def neighbor_features(
    samples: dict[str, np.ndarray],
    context: dict[str, object],
    train_compact: np.ndarray,
    train_pdp: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(samples["sample_id"])
    names = (
        "mass_mean", "mass_std", "mass_median", "mass_q10", "mass_q90", "mass_max",
        "mass_cv", "near_mass", "far_mass", "near_minus_far", "weighted_mass",
        "distance_slope", "support_fraction", "near_support", "far_support",
        "near_minus_far_support", "angle_valid_fraction", "angle_h_std", "angle_v_std",
        "angle_offset_norm_mean", "quadrant_mass_std", "quadrant_mass_range",
        "quadrant_support_count", "pdp_dispersion", "pdp_delay_concentration",
        "pdp_top1_mean", "pdp_top1_std", "pdp_entropy_mean", "pdp_entropy_std",
    )
    output = {f"neighbor_{name}": np.empty(count, dtype=np.float32) for name in names}
    peak_mask = e169.build_peak_mask(train_compact)
    train_positions = np.asarray(context["train_positions"], dtype=np.float64)
    query_ids = np.asarray(context["query_ids"])
    donor_all = np.asarray(context["neighbor_ids"])
    distance_all = np.asarray(context["neighbor_distances"], dtype=np.float64)
    pdp = np.maximum(np.asarray(train_pdp, dtype=np.float32), 0.0)
    pdp /= np.maximum(pdp.sum(axis=1, keepdims=True), TINY)
    pdp_bins = np.argmax(pdp, axis=1)
    pdp_top1 = np.max(pdp, axis=1)
    pdp_entropy = np.asarray([entropy(row) for row in pdp], dtype=np.float32)
    for index in range(count):
        point = int(samples["oof_row"][index])
        ue = int(samples["ue"][index])
        h = int(samples["h"][index])
        v = int(samples["v"][index])
        donors = donor_all[point]
        distances = distance_all[point]
        maps = train_compact[donors, ue]
        masses = e169.window_mass_batch(maps, h, v)
        support = e169.support_near_coordinate(peak_mask[donors, ue], h, v)
        weights = 1.0 / np.square(distances + 2.0)
        design = np.column_stack([np.ones(len(distances)), distances - np.mean(distances)])
        coefficient = np.linalg.lstsq(design * np.sqrt(weights[:, None]), masses * np.sqrt(weights), rcond=None)[0]
        offsets, valid = e169.local_peak_offsets(maps, peak_mask[donors, ue], h, v)
        query_xy = train_positions[query_ids[point], :2]
        delta = train_positions[donors, :2] - query_xy
        quadrant = np.floor(np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2.0 * np.pi) / (0.5 * np.pi)).astype(np.int64)
        quadrant_mass = []
        quadrant_support = []
        for q in range(4):
            selected = quadrant == q
            if np.any(selected):
                quadrant_mass.append(float(np.mean(masses[selected])))
                quadrant_support.append(float(np.mean(support[selected])))
        donor_pdp = pdp[donors]
        values = {
            "mass_mean": np.mean(masses),
            "mass_std": np.std(masses),
            "mass_median": np.median(masses),
            "mass_q10": np.quantile(masses, 0.10),
            "mass_q90": np.quantile(masses, 0.90),
            "mass_max": np.max(masses),
            "mass_cv": np.std(masses) / max(np.mean(masses), 1e-6),
            "near_mass": np.mean(masses[:NEAR_COUNT]),
            "far_mass": np.mean(masses[FAR_START:]),
            "near_minus_far": np.mean(masses[:NEAR_COUNT]) - np.mean(masses[FAR_START:]),
            "weighted_mass": np.sum(weights * masses) / np.sum(weights),
            "distance_slope": coefficient[1],
            "support_fraction": np.mean(support),
            "near_support": np.mean(support[:NEAR_COUNT]),
            "far_support": np.mean(support[FAR_START:]),
            "near_minus_far_support": np.mean(support[:NEAR_COUNT]) - np.mean(support[FAR_START:]),
            "angle_valid_fraction": np.mean(valid),
            "angle_h_std": np.std(offsets[valid, 0]) if np.any(valid) else 4.0,
            "angle_v_std": np.std(offsets[valid, 1]) if np.any(valid) else 2.0,
            "angle_offset_norm_mean": np.mean(np.linalg.norm(offsets[valid], axis=1)) if np.any(valid) else 4.0,
            "quadrant_mass_std": np.std(quadrant_mass) if quadrant_mass else 0.0,
            "quadrant_mass_range": np.ptp(quadrant_mass) if quadrant_mass else 0.0,
            "quadrant_support_count": np.sum(np.asarray(quadrant_support) >= 0.5),
            "pdp_dispersion": cosine_dispersion(donor_pdp),
            "pdp_delay_concentration": circular_concentration(pdp_bins[donors], 192, pdp_top1[donors]),
            "pdp_top1_mean": np.mean(pdp_top1[donors]),
            "pdp_top1_std": np.std(pdp_top1[donors]),
            "pdp_entropy_mean": np.mean(pdp_entropy[donors]),
            "pdp_entropy_std": np.std(pdp_entropy[donors]),
        }
        for name, value in values.items():
            output[f"neighbor_{name}"][index] = value
        if (index + 1) % 1000 == 0 or index + 1 == count:
            print(f"E171 neighbor peak features: {index + 1}/{count}", flush=True)
    return output


def cross_ue_features(samples: dict[str, np.ndarray], v7: np.ndarray) -> dict[str, np.ndarray]:
    count = len(samples["sample_id"])
    masses = np.empty((count, 4), dtype=np.float32)
    peak_support = np.empty((count, 4), dtype=np.float32)
    masks = e169.build_peak_mask(v7)
    for index in range(count):
        point = int(samples["oof_row"][index])
        h = int(samples["h"][index])
        v = int(samples["v"][index])
        for ue in range(4):
            masses[index, ue] = e169.window_mass(v7[point, ue], h, v)
            peak_support[index, ue] = e169.support_near_coordinate(masks[point, ue][None], h, v)[0]
    current = masses[np.arange(count), samples["ue"]]
    columns: dict[str, np.ndarray] = {}
    add_feature(columns, "context_ue_mass_mean", np.mean(masses, axis=1))
    add_feature(columns, "context_ue_mass_std", np.std(masses, axis=1))
    add_feature(columns, "context_ue_mass_min", np.min(masses, axis=1))
    add_feature(columns, "context_ue_mass_max", np.max(masses, axis=1))
    add_feature(columns, "context_ue_current_to_mean", current / np.maximum(np.mean(masses, axis=1), 1e-6))
    add_feature(columns, "context_ue_peak_support_count", np.sum(peak_support, axis=1))
    return columns


def geometry_features(samples: dict[str, np.ndarray], context: dict[str, object]) -> dict[str, np.ndarray]:
    count = len(samples["sample_id"])
    columns: dict[str, np.ndarray] = {}
    point_rows = context["point_rows"]
    query_ids = np.asarray(context["query_ids"])
    train_positions = np.asarray(context["train_positions"], dtype=np.float64)
    setup = context["context"]["setup"]
    map_cache = np.load(MAP_FEATURES, allow_pickle=False)
    map_names = [str(item) for item in map_cache["feature_names"].tolist()]
    train_map = map_cache["train_features"].astype(np.float32)
    point_fields = (
        "normalized_hole_depth", "nearest_train_m", "knn8_distance_m",
        "angular_max_gap_deg", "one_sidedness", "quadrants_occupied",
        "neighbor_pas_dispersion", "neighbor_pdp_dispersion",
        "two_side_pas_difference", "two_side_pdp_difference",
        "neighbor_complex_h_variation_per_m",
    )
    for field in point_fields:
        values = []
        for point in samples["oof_row"]:
            text = point_rows[int(point)][field]
            values.append(float(text) if text.lower() != "nan" else float("nan"))
        add_feature(columns, f"geo_{field}", values)
    global_ids = query_ids[samples["oof_row"]]
    xy = train_positions[global_ids, :2]
    bs_xy = setup.bs_positions[samples["site"], :2]
    add_feature(columns, "geo_x", xy[:, 0])
    add_feature(columns, "geo_y", xy[:, 1])
    add_feature(columns, "geo_bs_distance", np.linalg.norm(xy - bs_xy, axis=1))
    add_feature(columns, "geo_site", samples["site"])
    for feature_index, name in enumerate(map_names):
        add_feature(columns, f"geo_map_{name}", train_map[global_ids, feature_index])
    return columns


def build_features(device: torch.device, batch_size: int, force: bool) -> Path:
    if FEATURE_CACHE.exists() and not force:
        print(f"E171 reuse feature cache: {PATHS.relative(FEATURE_CACHE)}", flush=True)
        return FEATURE_CACHE
    context = point_context()
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    v7_channel = np.load(V7_CHANNEL, mmap_mode="r")
    train_compact = e169.load_or_build_compact(
        train_channel, TRAIN_COMPACT, device, batch_size, rebuild=False
    )
    v7_compact = e169.load_or_build_compact(
        v7_channel, V7_COMPACT, device, batch_size, rebuild=False
    )
    truth_compact = np.asarray(train_compact[np.asarray(context["query_ids"])], dtype=np.float32)
    samples = build_peak_samples(v7_compact, truth_compact, context)
    by_point = sample_ids_by_point(samples)
    columns = self_features(samples, v7_compact)
    reference, reference_columns = compute_v7_reference(
        samples, v7_channel, by_point, device, batch_size
    )
    columns.update(reference_columns)
    columns.update(compute_carrier_features(samples, reference, by_point, device, batch_size))
    descriptors = np.load(SPECTRAL_DESCRIPTORS, allow_pickle=False)
    columns.update(
        neighbor_features(
            samples,
            context,
            train_compact,
            descriptors["global_pdp"].astype(np.float32),
        )
    )
    columns.update(cross_ue_features(samples, v7_compact))
    columns.update(geometry_features(samples, context))
    feature_names = np.asarray(list(columns), dtype="U96")
    matrix = np.column_stack([columns[name] for name in feature_names]).astype(np.float32)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURE_CACHE,
        X=matrix,
        feature_names=feature_names,
        **samples,
    )
    print(
        f"E171 feature cache: samples={len(samples['sample_id'])} features={matrix.shape[1]} "
        f"labeled={np.sum(samples['label'] >= 0)} suppress={np.sum(samples['label'] == SUPPRESS)}",
        flush=True,
    )
    return FEATURE_CACHE


def make_model(name: str, seed: int):
    seed = int(seed % (2**32 - 1))
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1600,
                solver="lbfgs",
                random_state=seed,
            ),
        )
    if name == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesClassifier(
                n_estimators=160,
                max_features=0.75,
                min_samples_leaf=6,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        )
    if name == "mlp":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=2e-3,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=180,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=12,
                random_state=seed,
            ),
        )
    raise ValueError(name)


def feature_indices(feature_names: np.ndarray, group: str) -> np.ndarray:
    prefixes = FEATURE_GROUPS[group]
    selected = [index for index, name in enumerate(feature_names.tolist()) if name.startswith(prefixes)]
    if not selected:
        raise RuntimeError(f"feature group {group} is empty")
    return np.asarray(selected, dtype=np.int64)


def fit_probability(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    unique = np.unique(y_train)
    if len(unique) < 2:
        return np.full(len(x_test), float(unique[0]), dtype=np.float64)
    finite_columns = np.any(np.isfinite(x_train), axis=0)
    if not np.any(finite_columns):
        return np.full(len(x_test), float(np.mean(y_train)), dtype=np.float64)
    x_train = x_train[:, finite_columns]
    x_test = x_test[:, finite_columns]
    model = make_model(model_name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def precision_frontier(
    labels: np.ndarray,
    probability: np.ndarray,
    error_mass: np.ndarray,
    candidate_mass: np.ndarray,
    target: float,
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    order = np.argsort(-p)
    y_sorted = y[order]
    p_sorted = p[order]
    error_sorted = error_mass[order]
    candidate_sorted = candidate_mass[order]
    tp = np.cumsum(y_sorted == SUPPRESS)
    count = np.arange(1, len(order) + 1)
    precision = tp / count
    total_positive = max(int(np.sum(y == SUPPRESS)), 1)
    total_mass = max(float(np.sum(error_mass[y == SUPPRESS])), TINY)
    captured_mass = np.cumsum(np.where(y_sorted == SUPPRESS, error_sorted, 0.0))
    selected_keep_mass = np.cumsum(np.where(y_sorted == KEEP, candidate_sorted, 0.0))
    mass_precision = captured_mass / np.maximum(captured_mass + selected_keep_mass, TINY)
    valid = np.flatnonzero(precision >= target)
    if len(valid) == 0:
        return {
            "threshold": float("inf"),
            "selected_count": 0,
            "precision": float("nan"),
            "event_coverage": 0.0,
            "mass_coverage": 0.0,
            "mass_precision": float("nan"),
        }
    best = int(valid[np.argmax(captured_mass[valid])])
    return {
        "threshold": float(p_sorted[best]),
        "selected_count": int(best + 1),
        "precision": float(precision[best]),
        "event_coverage": float(tp[best] / total_positive),
        "mass_coverage": float(captured_mass[best] / total_mass),
        "mass_precision": float(mass_precision[best]),
    }


def oof_probabilities(
    x: np.ndarray,
    labels: np.ndarray,
    holes: np.ndarray,
    model_name: str,
    seed_offset: int,
) -> np.ndarray:
    output = np.full(len(labels), np.nan, dtype=np.float64)
    for outer_index, outer in enumerate(sorted(np.unique(holes).tolist())):
        train = (holes != outer) & (labels >= 0)
        test = (holes == outer) & (labels >= 0)
        output[test] = fit_probability(
            model_name, x[train], labels[train], x[test], SEED + seed_offset + outer_index
        )
    return output


def evaluate_feature_groups(data: dict[str, np.ndarray]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    x = data["X"].astype(np.float32)
    names = data["feature_names"]
    labels = data["label"].astype(np.int64)
    holes = data["hole_id"]
    error_mass = data["suppress_error_mass"].astype(np.float64)
    candidate_mass = data["v7_mass"].astype(np.float64)
    valid = labels >= 0
    rows: list[dict[str, object]] = []
    hole_rows: list[dict[str, object]] = []
    probabilities: dict[tuple[str, str], np.ndarray] = {}
    for group_index, group in enumerate(FEATURE_GROUPS):
        indices = feature_indices(names, group)
        for model_index, model_name in enumerate(MODELS):
            probability = oof_probabilities(
                x[:, indices], labels, holes, model_name, 1000 * group_index + 100 * model_index
            )
            probabilities[(group, model_name)] = probability
            base = {
                "feature_group": group,
                "model": model_name,
                "feature_count": len(indices),
                "labeled_count": int(np.sum(valid)),
                "suppress_count": int(np.sum(labels[valid] == SUPPRESS)),
                "roc_auc": float(roc_auc_score(labels[valid], probability[valid])),
                "average_precision": float(average_precision_score(labels[valid], probability[valid])),
            }
            for target in TARGET_PRECISIONS:
                frontier = precision_frontier(
                    labels[valid], probability[valid], error_mass[valid], candidate_mass[valid], target
                )
                prefix = f"p{int(target * 100)}"
                for key, value in frontier.items():
                    base[f"{prefix}_{key}"] = value
            rows.append(base)
            p90 = float(base["p90_threshold"])
            for hole in sorted(np.unique(holes).tolist()):
                selected = valid & (holes == hole)
                release = selected & (probability >= p90) if math.isfinite(p90) else np.zeros(len(labels), dtype=bool)
                true = release & (labels == SUPPRESS)
                total_mass = float(np.sum(error_mass[selected & (labels == SUPPRESS)]))
                hole_rows.append(
                    {
                        "feature_group": group,
                        "model": model_name,
                        "hole_id": hole,
                        "labeled_count": int(np.sum(selected)),
                        "suppress_count": int(np.sum(selected & (labels == SUPPRESS))),
                        "selected_count": int(np.sum(release)),
                        "precision_at_global_p90_threshold": float(np.sum(true) / max(np.sum(release), 1)),
                        "suppress_mass_coverage_at_global_p90_threshold": float(np.sum(error_mass[true]) / max(total_mass, TINY)),
                    }
                )
            print(
                f"E171 OOF {group}/{model_name}: AP={base['average_precision']:.4f} "
                f"p90_mass={base['p90_mass_coverage']:.4f}",
                flush=True,
            )
    return rows, hole_rows, probabilities


def univariate_feature_audit(data: dict[str, np.ndarray]) -> list[dict[str, object]]:
    x = data["X"].astype(np.float32)
    names = data["feature_names"]
    labels = data["label"].astype(np.int64)
    holes = data["hole_id"]
    error_mass = data["suppress_error_mass"].astype(np.float64)
    candidate_mass = data["v7_mass"].astype(np.float64)
    valid = labels >= 0
    rows = []
    for index, name in enumerate(names.tolist()):
        probability = oof_probabilities(
            x[:, index : index + 1], labels, holes, "logistic", 50_000 + index * 20
        )
        frontier = precision_frontier(
            labels[valid], probability[valid], error_mass[valid], candidate_mass[valid], 0.90
        )
        rows.append(
            {
                "feature": name,
                "average_precision": float(average_precision_score(labels[valid], probability[valid])),
                "roc_auc": float(roc_auc_score(labels[valid], probability[valid])),
                "p90_selected_count": frontier["selected_count"],
                "p90_precision": frontier["precision"],
                "p90_event_coverage": frontier["event_coverage"],
                "p90_mass_coverage": frontier["mass_coverage"],
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(names):
            print(f"E171 univariate features: {index + 1}/{len(names)}", flush=True)
    return rows


def nested_release_masks(data: dict[str, np.ndarray]) -> tuple[list[dict[str, object]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    x = data["X"].astype(np.float32)
    indices = feature_indices(data["feature_names"], "all")
    x = x[:, indices]
    labels = data["label"].astype(np.int64)
    holes = data["hole_id"]
    error_mass = data["suppress_error_mass"].astype(np.float64)
    candidate_mass = data["v7_mass"].astype(np.float64)
    unique_holes = sorted(np.unique(holes).tolist())
    rows: list[dict[str, object]] = []
    masks: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    for model_index, model_name in enumerate(MODELS):
        release_all = np.zeros(len(labels), dtype=bool)
        probability_all = np.full(len(labels), np.nan, dtype=np.float64)
        for outer_index, outer in enumerate(unique_holes):
            inner_probability = np.full(len(labels), np.nan, dtype=np.float64)
            for inner_index, inner in enumerate(unique_holes):
                if inner == outer:
                    continue
                train = (holes != outer) & (holes != inner) & (labels >= 0)
                test = (holes == inner) & (labels >= 0)
                inner_probability[test] = fit_probability(
                    model_name,
                    x[train],
                    labels[train],
                    x[test],
                    SEED + 100_000 * model_index + 1000 * outer_index + inner_index,
                )
            inner_valid = (holes != outer) & (labels >= 0) & np.isfinite(inner_probability)
            frontier = precision_frontier(
                labels[inner_valid],
                inner_probability[inner_valid],
                error_mass[inner_valid],
                candidate_mass[inner_valid],
                0.90,
            )
            threshold = float(frontier["threshold"])
            train = (holes != outer) & (labels >= 0)
            test_all = holes == outer
            probability = fit_probability(
                model_name,
                x[train],
                labels[train],
                x[test_all],
                SEED + 200_000 * model_index + outer_index,
            )
            probability_all[test_all] = probability
            release = probability >= threshold if math.isfinite(threshold) else np.zeros(len(probability), dtype=bool)
            outer_ids = np.flatnonzero(test_all)
            release_all[outer_ids[release]] = True
            labeled_outer = test_all & (labels >= 0)
            released_labeled = release_all & labeled_outer
            true = released_labeled & (labels == SUPPRESS)
            total_mass = float(np.sum(error_mass[labeled_outer & (labels == SUPPRESS)]))
            rows.append(
                {
                    "model": model_name,
                    "hole_id": outer,
                    "inner_p90_threshold": threshold,
                    "inner_selected_count": frontier["selected_count"],
                    "inner_precision": frontier["precision"],
                    "inner_mass_coverage": frontier["mass_coverage"],
                    "outer_all_peak_count": int(np.sum(test_all)),
                    "outer_selected_all_count": int(np.sum(release_all & test_all)),
                    "outer_selected_ambiguous_count": int(np.sum(release_all & test_all & (labels < 0))),
                    "outer_selected_labeled_count": int(np.sum(released_labeled)),
                    "outer_precision": float(np.sum(true) / max(np.sum(released_labeled), 1)),
                    "outer_suppress_mass_coverage": float(np.sum(error_mass[true]) / max(total_mass, TINY)),
                }
            )
            print(
                f"E171 nested {model_name} {outer}: selected={np.sum(release_all & test_all)} "
                f"outer_precision={rows[-1]['outer_precision']:.3f}",
                flush=True,
            )
        masks[model_name] = release_all
        probabilities[model_name] = probability_all

    model_name = "adaptive_univariate_logistic"
    release_all = np.zeros(len(labels), dtype=bool)
    probability_all = np.full(len(labels), np.nan, dtype=np.float64)
    feature_names = data["feature_names"].tolist()
    for outer_index, outer in enumerate(unique_holes):
        best_feature = 0
        best_frontier: dict[str, float | int] | None = None
        best_key = (-1.0, -1.0, -1.0)
        for feature_index in range(x.shape[1]):
            inner_probability = np.full(len(labels), np.nan, dtype=np.float64)
            for inner_index, inner in enumerate(unique_holes):
                if inner == outer:
                    continue
                train = (holes != outer) & (holes != inner) & (labels >= 0)
                test = (holes == inner) & (labels >= 0)
                inner_probability[test] = fit_probability(
                    "logistic",
                    data["X"][train, feature_index : feature_index + 1],
                    labels[train],
                    data["X"][test, feature_index : feature_index + 1],
                    SEED + 300_000 + 10_000 * outer_index + 50 * feature_index + inner_index,
                )
            inner_valid = (holes != outer) & (labels >= 0) & np.isfinite(inner_probability)
            frontier = precision_frontier(
                labels[inner_valid],
                inner_probability[inner_valid],
                error_mass[inner_valid],
                candidate_mass[inner_valid],
                0.90,
            )
            average_precision = float(
                average_precision_score(labels[inner_valid], inner_probability[inner_valid])
            )
            key = (
                float(frontier["mass_coverage"]),
                float(frontier["event_coverage"]),
                average_precision,
            )
            if key > best_key:
                best_feature = feature_index
                best_frontier = frontier
                best_key = key
        if best_frontier is None:
            raise RuntimeError(f"no adaptive univariate feature selected for {outer}")
        threshold = float(best_frontier["threshold"])
        train = (holes != outer) & (labels >= 0)
        test_all = holes == outer
        probability = fit_probability(
            "logistic",
            data["X"][train, best_feature : best_feature + 1],
            labels[train],
            data["X"][test_all, best_feature : best_feature + 1],
            SEED + 400_000 + outer_index,
        )
        probability_all[test_all] = probability
        release = probability >= threshold if math.isfinite(threshold) else np.zeros(len(probability), dtype=bool)
        outer_ids = np.flatnonzero(test_all)
        release_all[outer_ids[release]] = True
        labeled_outer = test_all & (labels >= 0)
        released_labeled = release_all & labeled_outer
        true = released_labeled & (labels == SUPPRESS)
        total_mass = float(np.sum(error_mass[labeled_outer & (labels == SUPPRESS)]))
        rows.append(
            {
                "model": model_name,
                "hole_id": outer,
                "selected_feature": feature_names[best_feature],
                "inner_p90_threshold": threshold,
                "inner_selected_count": best_frontier["selected_count"],
                "inner_precision": best_frontier["precision"],
                "inner_mass_coverage": best_frontier["mass_coverage"],
                "outer_all_peak_count": int(np.sum(test_all)),
                "outer_selected_all_count": int(np.sum(release_all & test_all)),
                "outer_selected_ambiguous_count": int(np.sum(release_all & test_all & (labels < 0))),
                "outer_selected_labeled_count": int(np.sum(released_labeled)),
                "outer_precision": float(np.sum(true) / max(np.sum(released_labeled), 1)),
                "outer_suppress_mass_coverage": float(np.sum(error_mass[true]) / max(total_mass, TINY)),
            }
        )
        print(
            f"E171 nested {model_name} {outer}: feature={feature_names[best_feature]} "
            f"selected={np.sum(release_all & test_all)} outer_precision={rows[-1]['outer_precision']:.3f}",
            flush=True,
        )
    masks[model_name] = release_all
    probabilities[model_name] = probability_all
    return rows, masks, probabilities


def score_v7(truth: np.ndarray, query_ids: np.ndarray, v7: np.ndarray, batch_size: int) -> PointMetricStatistics:
    arrays = [np.empty(500, dtype=np.float64) for _ in range(4)]
    for start in range(0, 500, batch_size):
        stop = min(start + batch_size, 500)
        stats = point_metric_statistics(
            np.asarray(v7[start:stop], dtype=np.complex64),
            np.asarray(truth[query_ids[start:stop]], dtype=np.complex64),
        )
        for destination, source in zip(arrays, (stats.pas, stats.pdp, stats.squared_error, stats.truth_energy), strict=True):
            destination[start:stop] = source
    result = PointMetricStatistics(*arrays)
    expected = aggregate_statistics(result)
    if abs(float(expected["score"]) - e167.EXPECTED_V7["score"]) > e167.IDENTITY_TOLERANCE:
        raise RuntimeError(f"V7 identity failed before suppression: {expected}")
    return result


def suppress_and_score(
    v7: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    samples: dict[str, np.ndarray],
    release: np.ndarray,
    alpha: float,
    device: torch.device,
    batch_size: int,
) -> PointMetricStatistics:
    arrays = [np.empty(500, dtype=np.float64) for _ in range(4)]
    selected_ids = np.flatnonzero(release)
    by_point = [selected_ids[samples["oof_row"][selected_ids] == point] for point in range(500)]
    amplitude = math.sqrt(1.0 - alpha)
    for start in range(0, 500, batch_size):
        stop = min(start + batch_size, 500)
        value = torch.as_tensor(
            np.array(v7[start:stop], dtype=np.complex64, copy=True), device=device
        )
        reference_power = value.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
        with torch.inference_mode():
            shaped = value.reshape(len(value), 2, 16, 8, 4, 192)
            angle = torch.fft.fft2(shaped, dim=(2, 3))
            scale = torch.ones((len(value), 16, 8, 4), dtype=torch.float32, device=device)
            for point in range(start, stop):
                for sample_id in by_point[point]:
                    h = int(samples["h"][sample_id])
                    v = int(samples["v"][sample_id])
                    ue = int(samples["ue"][sample_id])
                    for hi in e169.angle_indices(h, 1, e169.N_H):
                        for vi in e169.angle_indices(v, 1, e169.N_V):
                            scale[point - start, hi, vi, ue] = min(
                                float(scale[point - start, hi, vi, ue]), amplitude
                            )
            angle *= scale[:, None, :, :, :, None]
            candidate = torch.fft.ifft2(angle, dim=(2, 3)).reshape_as(value)
            candidate_power = candidate.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
            candidate *= torch.sqrt(reference_power / candidate_power.clamp_min(TINY))[:, None, None, None]
        prediction = candidate.cpu().numpy().astype(np.complex64, copy=False)
        stats = point_metric_statistics(
            prediction,
            np.asarray(truth[query_ids[start:stop]], dtype=np.complex64),
        )
        for destination, source in zip(arrays, (stats.pas, stats.pdp, stats.squared_error, stats.truth_energy), strict=True):
            destination[start:stop] = source
    return PointMetricStatistics(*arrays)


def suppression_tables(
    data: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    context = point_context()
    query_ids = np.asarray(context["query_ids"])
    holes = np.asarray(context["hole_id"])
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    v7 = np.load(V7_CHANNEL, mmap_mode="r")
    baseline_stats = score_v7(truth, query_ids, v7, batch_size)
    baseline = aggregate_statistics(baseline_stats)
    samples = {key: data[key] for key in ("oof_row", "h", "v", "ue")}
    rows: list[dict[str, object]] = []
    hole_rows: list[dict[str, object]] = []
    for model_name, release in masks.items():
        for alpha in SUPPRESSION_LEVELS:
            statistics = suppress_and_score(
                v7, truth, query_ids, samples, release, alpha, device, batch_size
            )
            overall = aggregate_statistics(statistics)
            rows.append(
                {
                    "model": model_name,
                    "suppression_power_fraction": alpha,
                    "selected_peak_count": int(np.sum(release)),
                    **overall,
                    "delta_pas_vs_v7": float(overall["pas"]) - float(baseline["pas"]),
                    "delta_pdp_vs_v7": float(overall["pdp"]) - float(baseline["pdp"]),
                    "delta_nmse_vs_v7": float(overall["nmse"]) - float(baseline["nmse"]),
                    "delta_score_vs_v7": float(overall["score"]) - float(baseline["score"]),
                }
            )
            for hole in sorted(np.unique(holes).tolist()):
                mask = holes == hole
                current = aggregate_statistics(statistics, mask)
                base = aggregate_statistics(baseline_stats, mask)
                peak_mask = holes[data["oof_row"]] == hole
                hole_rows.append(
                    {
                        "model": model_name,
                        "suppression_power_fraction": alpha,
                        "hole_id": hole,
                        "selected_peak_count": int(np.sum(release & peak_mask)),
                        **current,
                        "delta_pas_vs_v7": float(current["pas"]) - float(base["pas"]),
                        "delta_pdp_vs_v7": float(current["pdp"]) - float(base["pdp"]),
                        "delta_nmse_vs_v7": float(current["nmse"]) - float(base["nmse"]),
                        "delta_score_vs_v7": float(current["score"]) - float(base["score"]),
                    }
                )
            print(
                f"E171 suppression {model_name} {alpha:.2f}: score={overall['score']:.9f} "
                f"delta={rows[-1]['delta_score_vs_v7']:+.9f}",
                flush=True,
            )
    return rows, hole_rows


def plot_precision_coverage(group_rows: list[dict[str, object]]) -> Path:
    selected = [row for row in group_rows if row["feature_group"] == "all"]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for row in selected:
        coverage = [float(row[f"p{int(target * 100)}_mass_coverage"]) for target in TARGET_PRECISIONS]
        ax.plot([80, 90, 95], coverage, marker="o", linewidth=2, label=str(row["model"]))
    ax.set_xlabel("事件precision目标 (%)")
    ax.set_ylabel("假峰错误质量coverage")
    ax.set_title("全部Clean特征：高precision下还能覆盖多少假峰质量")
    ax.set_xticks([80, 90, 95])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = VIS_DIR / "01_precision与假峰质量coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_groups(group_rows: list[dict[str, object]]) -> Path:
    labels = [f"{row['feature_group']}\n{row['model']}" for row in group_rows]
    values = [float(row["p90_mass_coverage"]) for row in group_rows]
    order = np.argsort(values)[::-1]
    take = order[: min(18, len(order))]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(np.arange(len(take)), np.asarray(values)[take][::-1], color="#2a9d8f")
    ax.set_yticks(np.arange(len(take)), np.asarray(labels)[take][::-1])
    ax.set_xlabel("90% precision下假峰错误质量coverage")
    ax.set_title("特征组与简单模型比较")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = VIS_DIR / "02_特征组90precision覆盖.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_suppression(rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for model in sorted({str(row["model"]) for row in rows}):
        selected = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: float(row["suppression_power_fraction"]),
        )
        ax.plot(
            [100 * float(row["suppression_power_fraction"]) for row in selected],
            [float(row["delta_score_vs_v7"]) for row in selected],
            marker="o",
            linewidth=2,
            label=model,
        )
    ax.axhline(0.0, color="#444444", linewidth=1, linestyle="--")
    ax.set_xlabel("选中峰的功率压低比例 (%)")
    ax.set_ylabel("fixed500完整总分相对V7")
    ax.set_title("高置信SUPPRESS峰能否换成完整分数")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = VIS_DIR / "03_保守压峰完整分变化.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_report(report: dict[str, object], figures: list[Path]) -> None:
    label = report["label_summary"]
    best_rows = sorted(
        report["feature_group_results"],
        key=lambda row: float(row["p90_mass_coverage"]),
        reverse=True,
    )[:12]
    nested = report["nested_selection_summary"]
    suppression = sorted(
        report["suppression_results"],
        key=lambda row: float(row["delta_score_vs_v7"]),
        reverse=True,
    )
    best_suppression = suppression[0]
    feature_rows = sorted(
        report["univariate_feature_results"],
        key=lambda row: float(row["p90_mass_coverage"]),
        reverse=True,
    )
    lines = [
        "# V7假峰能否靠Clean信息露馅",
        "",
        "本轮只使用官方H、坐标、Setup、地图、Clean V7和17种官方H复载体。没有读取任何Sionna或V8-V12派生量。",
        "",
        "## 标签",
        "",
        f"V7峰总数=`{label['peak_count']}`，KEEP=`{label['keep_count']}`，SUPPRESS=`{label['suppress_count']}`，丢弃边界=`{label['ambiguous_count']}`。",
        "",
        "## 特征组与模型",
        "",
        "| 特征组 | 模型 | AP | 90% precision实际值 | 事件coverage | 假峰质量coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['feature_group']} | {row['model']} | {row['average_precision']:.4f} | "
            f"{row['p90_precision']:.1%} | {row['p90_event_coverage']:.1%} | {row['p90_mass_coverage']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 严格嵌套90% precision选择",
            "",
            "| 模型 | 选中峰 | 标注峰precision | 假峰质量coverage | 边界峰数 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in nested:
        lines.append(
            f"| {row['model']} | {row['selected_all_count']} | {row['precision']:.1%} | "
            f"{row['mass_coverage']:.1%} | {row['selected_ambiguous_count']} |"
        )
    lines.extend(
        [
            "",
            "## 保守压峰完整官方分",
            "",
            "| 模型 | 压低 | PAS | PDP | NMSE | 总分 | 相对V7 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in suppression:
        lines.append(
            f"| {row['model']} | {row['suppression_power_fraction']:.0%} | "
            f"{row['pas']:.6f} | {row['pdp']:.6f} | {row['nmse']:.6f} | "
            f"{row['score']:.6f} | {row['delta_score_vs_v7']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 单特征",
            "",
            "90% precision质量coverage最高的10项：",
            "",
            "| 特征 | AP | 90%质量coverage |",
            "|---|---:|---:|",
        ]
    )
    for row in feature_rows[:10]:
        lines.append(f"| {row['feature']} | {row['average_precision']:.4f} | {row['p90_mass_coverage']:.1%} |")
    lines.extend(["", "基本无信息的10项：", "", "| 特征 | AP | 90%质量coverage |", "|---|---:|---:|"])
    for row in feature_rows[-10:]:
        lines.append(f"| {row['feature']} | {row['average_precision']:.4f} | {row['p90_mass_coverage']:.1%} |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            str(report["conclusion"]),
            "",
            f"完整分最优动作=`{best_suppression['model']}`压低`{best_suppression['suppression_power_fraction']:.0%}`，相对V7=`{best_suppression['delta_score_vs_v7']:+.6f}`。",
            "",
            "## 图",
            "",
        ]
    )
    for path in figures:
        relative = Path("../../visualizations/复赛V7假峰Clean高精度压峰审计") / path.name
        lines.extend([f"![{path.stem}]({relative.as_posix()})", ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def audit(device: torch.device, batch_size: int) -> None:
    data_file = np.load(FEATURE_CACHE, allow_pickle=False)
    data = {key: data_file[key] for key in data_file.files}
    labels = data["label"].astype(np.int64)
    label_summary = {
        "peak_count": len(labels),
        "keep_count": int(np.sum(labels == KEEP)),
        "suppress_count": int(np.sum(labels == SUPPRESS)),
        "ambiguous_count": int(np.sum(labels == AMBIGUOUS)),
        "labeled_suppress_fraction": float(np.mean(labels[labels >= 0] == SUPPRESS)),
        "total_suppress_error_mass": float(np.sum(data["suppress_error_mass"])),
    }
    group_rows, hole_classification_rows, group_probabilities = evaluate_feature_groups(data)
    feature_rows = univariate_feature_audit(data)
    nested_rows, release_masks, nested_probabilities = nested_release_masks(data)
    suppression_rows, suppression_hole_rows = suppression_tables(
        data, release_masks, device, batch_size
    )
    nested_summary = []
    for model in release_masks:
        selected = release_masks[model]
        labeled_selected = selected & (labels >= 0)
        true = labeled_selected & (labels == SUPPRESS)
        total_mass = max(float(np.sum(data["suppress_error_mass"][labels == SUPPRESS])), TINY)
        nested_summary.append(
            {
                "model": model,
                "selected_all_count": int(np.sum(selected)),
                "selected_labeled_count": int(np.sum(labeled_selected)),
                "selected_ambiguous_count": int(np.sum(selected & (labels < 0))),
                "precision": float(np.sum(true) / max(np.sum(labeled_selected), 1)),
                "event_coverage": float(np.sum(true) / max(np.sum(labels == SUPPRESS), 1)),
                "mass_coverage": float(np.sum(data["suppress_error_mass"][true]) / total_mass),
            }
        )
    prediction_rows = []
    all_probabilities = {model: group_probabilities[("all", model)] for model in MODELS}
    for index in range(len(labels)):
        row: dict[str, object] = {
            key: data[key][index].item() if hasattr(data[key][index], "item") else data[key][index]
            for key in (
                "sample_id", "oof_row", "global_train_id", "hole_id", "site", "ue", "h", "v",
                "rank", "v7_peak_value", "v7_mass", "gt_mass", "nearest_gt_peak_distance",
                "label", "suppress_error_mass",
            )
        }
        for model in MODELS:
            row[f"{model}_all_features_oof_probability"] = all_probabilities[model][index]
        for model in release_masks:
            row[f"{model}_nested_probability"] = nested_probabilities[model][index]
            row[f"{model}_nested_release"] = bool(release_masks[model][index])
        prediction_rows.append(row)
    write_csv(LABEL_CSV, [label_summary])
    write_csv(GROUP_CSV, group_rows)
    write_csv(HOLE_CLASSIFICATION_CSV, hole_classification_rows)
    write_csv(FEATURE_CSV, feature_rows)
    write_csv(NESTED_CSV, nested_rows)
    write_csv(SUPPRESSION_CSV, suppression_rows)
    write_csv(SUPPRESSION_HOLE_CSV, suppression_hole_rows)
    write_csv(PEAK_PREDICTION_CSV, prediction_rows)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_precision_coverage(group_rows),
        plot_feature_groups(group_rows),
        plot_suppression(suppression_rows),
    ]
    best_group = max(group_rows, key=lambda row: float(row["p90_mass_coverage"]))
    best_univariate = max(feature_rows, key=lambda row: float(row["p90_mass_coverage"]))
    best_suppression = max(suppression_rows, key=lambda row: float(row["delta_score_vs_v7"]))
    strict_candidates = [row for row in nested_summary if float(row["precision"]) >= 0.90]
    if strict_candidates:
        best_strict = max(strict_candidates, key=lambda row: float(row["mass_coverage"]))
        relation = (
            f"严格未见洞达到90% precision，假峰质量coverage="
            f"{float(best_strict['mass_coverage']):.3%}"
        )
    else:
        relation = (
            f"汇总OOF单特征最高90% precision质量coverage="
            f"{float(best_univariate['p90_mass_coverage']):.3%}，但严格未见洞没有任何方法保持90% precision"
        )
    if float(best_suppression["delta_score_vs_v7"]) >= 1e-3:
        action = "保守压峰已转化为有实质意义的fixed500完整分"
    elif float(best_suppression["delta_score_vs_v7"]) > 0.0:
        action = "最优完整分只有浮点量级微升，不能视为有效提分"
    else:
        action = "分类信号没有转化为正的fixed500完整分，压峰接口判定无用"
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "clean_contract": {
            "sionna_used": False,
            "v8_v12_used": False,
            "official_inputs_only": True,
            "validation": "10-hole leave-one-hole-out; nested 9-hole threshold calibration",
        },
        "label_summary": label_summary,
        "feature_group_results": group_rows,
        "per_hole_classification": hole_classification_rows,
        "univariate_feature_results": feature_rows,
        "nested_selection_per_hole": nested_rows,
        "nested_selection_summary": nested_summary,
        "suppression_results": suppression_rows,
        "suppression_per_hole": suppression_hole_rows,
        "conclusion": f"{relation}；{action}。",
        "outputs": [
            str(PATHS.relative(path))
            for path in (
                REPORT_MD, LABEL_CSV, GROUP_CSV, HOLE_CLASSIFICATION_CSV, FEATURE_CSV,
                NESTED_CSV, SUPPRESSION_CSV, SUPPRESSION_HOLE_CSV, PEAK_PREDICTION_CSV,
                *figures,
            )
        ],
    }
    REPORT_JSON.write_text(
        json.dumps(json_compatible(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_report(report, figures)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "CPU"
    print(f"E171 phase={args.phase} device={device} gpu={gpu_name}", flush=True)
    if args.phase in {"features", "all"}:
        build_features(device, args.batch_size, args.force_features)
    if args.phase in {"audit", "all"}:
        if not FEATURE_CACHE.exists():
            raise FileNotFoundError(FEATURE_CACHE)
        e167.configure_chinese_font()
        audit(device, args.batch_size)
    print(f"E171 phase {args.phase} complete in {time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
