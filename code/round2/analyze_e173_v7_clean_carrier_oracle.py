#!/usr/bin/env python3
"""Audit V7's 17 Clean complex carriers and their GT selection headroom."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

import analyze_e167_clean_test_hole_difficulty as e167
import analyze_e169_hard_hole_pas_peak_precursors as e169
from common.official_metrics import PointMetricStatistics, aggregate_statistics
from data import load_positions
from evaluate_v3_anchored_projection import TARGET_AD, TARGET_PAS, TARGET_PDP, target_batch
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META
from run_v4_calibrated_exact_projection import (
    OOF_CALIBRATED_PAS,
    OOF_CALIBRATED_PDP,
    SELECTION as V4_SELECTION,
)
from run_v6_regime_component_lifecycle import (
    REGIME_CACHE,
    REGIME_NAMES,
    TARGET_PATHS,
    _candidate_from_name,
    _parse_v4_candidate,
    apply_candidate,
)
from run_v7_complex_carrier_exact_projection import fit_scales


VERSION = "Round2-E173-v7-clean-carrier-oracle"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e173_v7_clean_carrier_oracle"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e173_v7_clean_carrier_oracle"
VIS_DIR = PATHS.analysis / "visualizations" / "复赛V7十七路Clean复载体Oracle审计"

CARRIER_BANK = PATHS.artifacts / "round2" / "complex_carrier_bank" / "oof_complex_carrier_bank.npy"
CARRIER_META = PATHS.artifacts / "round2" / "complex_carrier_bank" / "complex_carrier_bank_metadata.json"
GRAM_CACHE = PATHS.artifacts / "round2" / "complex_carrier_stacking" / "oof_port_gram.npz"
RAW_STACK = PATHS.artifacts / "round2" / "complex_carrier_stacking" / "oof_stacked_complex_channel.npy"
PROJECTED_STACK = PATHS.artifacts / "round2" / "complex_carrier_stacking" / "oof_stacked_v4_projection.npy"
SCALED_PROJECTED_STACK = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_lifecycle_overlay"
    / "oof_stacked_v4_projection_nested_scaled.npy"
)
LIFECYCLE_SELECTION = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_lifecycle_overlay"
    / "stacked_lifecycle_selection.json"
)
V7_CHANNEL = e167.V7_COMPLETE
V12_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_frozen_geometry_sequence_oof_channel.npy"
)
TRAIN_CHANNEL = e167.TRAIN_CHANNEL
TRAIN_COMPACT = e169.TRAIN_COMPACT
V7_COMPACT = e169.V7_COMPACT
V12_COMPACT = (
    PATHS.artifacts
    / "round2"
    / "e170_v7_v12_complete_hole_audit"
    / "V12_compact_pas.npy"
)

METRIC_CACHE = ARTIFACT_DIR / "carrier_point_metrics_and_compact.npz"
WEIGHT_CACHE = ARTIFACT_DIR / "gt_oracle_weights_and_selections.npz"
REPORT_JSON = ANALYSIS_DIR / "v7_clean_carrier_oracle.json"
REPORT_MD = ANALYSIS_DIR / "V7十七路Clean答案到底有没有正确候选.md"
OVERALL_CSV = ANALYSIS_DIR / "17载体与V7各阶段完整官方分.csv"
HOLE_CSV = ANALYSIS_DIR / "17载体十洞官方分.csv"
PAIR_CSV = ANALYSIS_DIR / "载体PAS错误相关性.csv"
ORACLE_CSV = ANALYSIS_DIR / "GT_oracle完整500点官方分.csv"
ORACLE_HOLE_CSV = ANALYSIS_DIR / "GT_oracle十洞官方分.csv"
SELECTION_CSV = ANALYSIS_DIR / "逐点与逐洞oracle选择统计.csv"
HARD_HOLE_CSV = ANALYSIS_DIR / "四个困难洞最佳载体与V7差距.csv"
V12_CSV = ANALYSIS_DIR / "V12修对点中Clean载体候选存在率.csv"
WEIGHT_CSV = ANALYSIS_DIR / "凸组合权重统计.csv"
FUSION_CSV = ANALYSIS_DIR / "已有载体优于融合结果的逐点统计.csv"

N_CARRIER = 17
N_POINT = 500
N_UE = 4
N_POL = 2
N_H = 16
N_V = 8
N_SC = 192
TINY = 1e-30
SEED = 20_260_811_173
EXPECTED_POINT_NMSE_ORACLE = 0.8936318883291309
EXPECTED_V7 = e167.EXPECTED_V7
EXPECTED_V12 = {
    "pas": 0.6625825788,
    "pdp": 0.7727009798,
    "nmse": 0.9340978586,
    "score": 0.6775208075,
}
HARD_HOLES = ("P_S1F1", "P_S1F2", "P_S1F3", "P_S1F4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force-metrics", action="store_true")
    parser.add_argument("--force-weights", action="store_true")
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


def point_metrics_torch(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = len(prediction)
    pred_shaped = prediction.reshape(batch, N_POL, N_H, N_V, N_UE, N_SC)
    truth_shaped = truth.reshape(batch, N_POL, N_H, N_V, N_UE, N_SC)
    pred_pas = torch.fft.fft2(pred_shaped, dim=(2, 3)).abs().square().float()
    truth_pas = torch.fft.fft2(truth_shaped, dim=(2, 3)).abs().square().float()
    pred_pas_flat = pred_pas.reshape(batch, 256, N_UE, N_SC)
    truth_pas_flat = truth_pas.reshape(batch, 256, N_UE, N_SC)
    pas_num = (pred_pas_flat * truth_pas_flat).sum(dim=1, dtype=torch.float64)
    pas_den = torch.sqrt(
        pred_pas_flat.square().sum(dim=1, dtype=torch.float64)
        * truth_pas_flat.square().sum(dim=1, dtype=torch.float64)
    )
    pas_den = torch.where(pas_den == 0.0, torch.full_like(pas_den, 1e-12), pas_den)
    pas = (pas_num / pas_den).mean(dim=(1, 2), dtype=torch.float64)

    pred_pdp = torch.fft.ifft(prediction, dim=-1).abs().square().float()
    truth_pdp = torch.fft.ifft(truth, dim=-1).abs().square().float()
    pdp_num = (pred_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_den = torch.sqrt(
        pred_pdp.square().sum(dim=3, dtype=torch.float64)
        * truth_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp_den = torch.where(pdp_den == 0.0, torch.full_like(pdp_den, 1e-12), pdp_den)
    pdp = (pdp_num / pdp_den).mean(dim=(1, 2), dtype=torch.float64)
    error = (prediction - truth).abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)

    compact = pred_pas.sum(dim=1).mean(dim=-1).permute(0, 3, 1, 2)
    compact_total = compact.sum(dim=(2, 3), keepdim=True)
    compact = torch.where(
        compact_total > 0.0,
        compact / compact_total.clamp_min(TINY),
        torch.zeros_like(compact),
    )
    return (
        pas.cpu().numpy(),
        pdp.cpu().numpy(),
        error.cpu().numpy(),
        energy.cpu().numpy(),
        compact.cpu().numpy().astype(np.float32),
    )


def statistics_from_arrays(
    pas: np.ndarray, pdp: np.ndarray, error: np.ndarray, energy: np.ndarray
) -> PointMetricStatistics:
    return PointMetricStatistics(
        np.asarray(pas, dtype=np.float64),
        np.asarray(pdp, dtype=np.float64),
        np.asarray(error, dtype=np.float64),
        np.asarray(energy, dtype=np.float64),
    )


def aggregate_arrays(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    return aggregate_statistics(statistics_from_arrays(pas, pdp, error, energy), mask)


def load_names() -> list[str]:
    metadata = json.loads(CARRIER_META.read_text(encoding="utf-8"))
    names = [str(name) for name in metadata["config_names"]]
    if len(names) != N_CARRIER:
        raise RuntimeError(f"expected {N_CARRIER} carrier names, got {len(names)}")
    if not bool(metadata.get("official_inputs_only")) or bool(metadata.get("sionna_used")):
        raise RuntimeError("carrier bank metadata failed Clean lineage gate")
    return names


def build_carrier_metrics(
    device: torch.device, batch_size: int, force: bool
) -> dict[str, np.ndarray]:
    required = {
        "pas": (N_CARRIER, N_POINT),
        "pdp": (N_CARRIER, N_POINT),
        "error": (N_CARRIER, N_POINT),
        "energy": (N_POINT,),
        "compact_pas": (N_CARRIER, N_POINT, N_UE, N_H, N_V),
    }
    if METRIC_CACHE.is_file() and not force:
        cache = np.load(METRIC_CACHE, allow_pickle=False)
        if all(key in cache and cache[key].shape == shape for key, shape in required.items()):
            return {key: cache[key] for key in required}

    bank = np.load(CARRIER_BANK, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    if bank.shape != (N_CARRIER, N_POINT, 256, N_UE, N_SC):
        raise ValueError(f"invalid carrier bank shape: {bank.shape}")
    output = {
        "pas": np.empty((N_CARRIER, N_POINT), dtype=np.float64),
        "pdp": np.empty((N_CARRIER, N_POINT), dtype=np.float64),
        "error": np.empty((N_CARRIER, N_POINT), dtype=np.float64),
        "energy": np.empty(N_POINT, dtype=np.float64),
        "compact_pas": np.empty((N_CARRIER, N_POINT, N_UE, N_H, N_V), dtype=np.float16),
    }
    for carrier in range(N_CARRIER):
        for start in range(0, N_POINT, batch_size):
            end = min(start + batch_size, N_POINT)
            prediction = torch.as_tensor(
                np.array(bank[carrier, start:end], dtype=np.complex64, copy=True),
                device=device,
            )
            truth = torch.as_tensor(
                np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
                device=device,
            )
            with torch.inference_mode():
                pas, pdp, error, energy, compact = point_metrics_torch(prediction, truth)
            output["pas"][carrier, start:end] = pas
            output["pdp"][carrier, start:end] = pdp
            output["error"][carrier, start:end] = error
            output["energy"][start:end] = energy
            output["compact_pas"][carrier, start:end] = compact.astype(np.float16)
        print(f"E173 carrier metrics {carrier + 1}/{N_CARRIER}", flush=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(METRIC_CACHE, **output)
    return output


def score_channel(
    channel: np.ndarray,
    device: torch.device,
    batch_size: int,
    query_ids: np.ndarray,
) -> dict[str, np.ndarray | dict[str, float | int]]:
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    pas = np.empty(N_POINT, dtype=np.float64)
    pdp = np.empty(N_POINT, dtype=np.float64)
    error = np.empty(N_POINT, dtype=np.float64)
    energy = np.empty(N_POINT, dtype=np.float64)
    compact = np.empty((N_POINT, N_UE, N_H, N_V), dtype=np.float32)
    for start in range(0, N_POINT, batch_size):
        end = min(start + batch_size, N_POINT)
        prediction = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        with torch.inference_mode():
            values = point_metrics_torch(prediction, truth)
        pas[start:end], pdp[start:end], error[start:end], energy[start:end], compact[start:end] = values
    return {
        "pas": pas,
        "pdp": pdp,
        "error": error,
        "energy": energy,
        "compact_pas": compact,
        "overall": aggregate_arrays(pas, pdp, error, energy),
    }


def baseline_metrics(
    device: torch.device, batch_size: int, query_ids: np.ndarray
) -> dict[str, dict[str, object]]:
    paths = {
        "raw_ridge_stack": RAW_STACK,
        "projected_stack_unscaled": PROJECTED_STACK,
        "projected_stack_nested_scaled": SCALED_PROJECTED_STACK,
        "V7_complete": V7_CHANNEL,
        "V12_offline_reference": V12_CHANNEL,
    }
    output: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        channel = np.load(path, mmap_mode="r")
        if channel.shape != (N_POINT, 256, N_UE, N_SC):
            raise ValueError(f"invalid baseline {name}: {channel.shape}")
        output[name] = score_channel(channel, device, batch_size, query_ids)
        print(f"E173 baseline metrics {name}", flush=True)
    for key, expected in EXPECTED_V7.items():
        error = abs(float(output["V7_complete"]["overall"][key]) - expected)
        if error > 5e-6:
            raise RuntimeError(f"V7 identity gate failed for {key}: {error}")
    for key, expected in EXPECTED_V12.items():
        error = abs(float(output["V12_offline_reference"]["overall"][key]) - expected)
        if error > 5e-6:
            raise RuntimeError(f"V12 identity gate failed for {key}: {error}")
    return output


def canonical_holes() -> tuple[list[e167.Hole], np.ndarray]:
    _, holes, _ = e167.build_holes()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    global_to_oof = np.full(4000, -1, dtype=np.int64)
    global_to_oof[query_ids] = np.arange(N_POINT, dtype=np.int64)
    return holes, global_to_oof


def carrier_tables(
    names: list[str],
    metrics: dict[str, np.ndarray],
    baselines: dict[str, dict[str, object]],
    holes: list[e167.Hole],
    global_to_oof: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    overall_rows: list[dict[str, object]] = []
    hole_rows: list[dict[str, object]] = []
    for carrier, name in enumerate(names):
        overall = aggregate_arrays(
            metrics["pas"][carrier],
            metrics["pdp"][carrier],
            metrics["error"][carrier],
            metrics["energy"],
        )
        overall_rows.append({"kind": "carrier", "name": name, **overall})
        for hole in holes:
            rows = global_to_oof[hole.point_ids]
            mask = np.zeros(N_POINT, dtype=bool)
            mask[rows] = True
            hole_rows.append(
                {
                    "kind": "carrier",
                    "name": name,
                    "hole_id": hole.hole_id,
                    "site": hole.site,
                    "fold": hole.ordinal,
                    **aggregate_arrays(
                        metrics["pas"][carrier],
                        metrics["pdp"][carrier],
                        metrics["error"][carrier],
                        metrics["energy"],
                        mask,
                    ),
                }
            )
    for name, values in baselines.items():
        overall_rows.append({"kind": "baseline", "name": name, **values["overall"]})
        for hole in holes:
            rows = global_to_oof[hole.point_ids]
            mask = np.zeros(N_POINT, dtype=bool)
            mask[rows] = True
            hole_rows.append(
                {
                    "kind": "baseline",
                    "name": name,
                    "hole_id": hole.hole_id,
                    "site": hole.site,
                    "fold": hole.ordinal,
                    **aggregate_arrays(
                        values["pas"], values["pdp"], values["error"], values["energy"], mask
                    ),
                }
            )
    return overall_rows, hole_rows


def residual_correlations(
    names: list[str], metrics: dict[str, np.ndarray], query_ids: np.ndarray
) -> tuple[list[dict[str, object]], dict[str, object]]:
    truth_compact_all = np.load(TRAIN_COMPACT, mmap_mode="r")
    truth = np.asarray(truth_compact_all[query_ids], dtype=np.float32)
    root_truth = np.sqrt(np.maximum(truth, 0.0))
    residual = np.sqrt(np.maximum(np.asarray(metrics["compact_pas"], dtype=np.float32), 0.0)) - root_truth[None]
    flat = residual.reshape(N_CARRIER, -1).astype(np.float64)
    pearson = np.corrcoef(flat)
    norm = np.linalg.norm(flat, axis=1)
    cosine = flat @ flat.T / np.maximum(norm[:, None] * norm[None, :], TINY)
    point_error = 1.0 - np.asarray(metrics["pas"], dtype=np.float64)
    point_error_corr = np.corrcoef(point_error)
    rows: list[dict[str, object]] = []
    values = []
    for left in range(N_CARRIER):
        for right in range(left + 1, N_CARRIER):
            row = {
                "left": names[left],
                "right": names[right],
                "compact_root_residual_pearson": float(pearson[left, right]),
                "compact_root_residual_cosine": float(cosine[left, right]),
                "point_pas_error_pearson": float(point_error_corr[left, right]),
            }
            rows.append(row)
            values.append(row)
    return rows, {
        "compact_root_residual_pearson_mean": float(np.mean([row["compact_root_residual_pearson"] for row in values])),
        "compact_root_residual_pearson_median": float(np.median([row["compact_root_residual_pearson"] for row in values])),
        "compact_root_residual_pearson_min": float(np.min([row["compact_root_residual_pearson"] for row in values])),
        "compact_root_residual_pearson_max": float(np.max([row["compact_root_residual_pearson"] for row in values])),
        "point_pas_error_pearson_mean": float(np.mean([row["point_pas_error_pearson"] for row in values])),
        "pearson_matrix": pearson,
        "cosine_matrix": cosine,
        "point_error_matrix": point_error_corr,
        "compact_residual": residual,
    }


def exact_point_score_selection(metrics: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, object]]:
    total_energy = float(np.sum(metrics["energy"]))
    spectral = 0.4 * (metrics["pas"] + metrics["pdp"]) / N_POINT
    normalized_error = metrics["error"] / max(total_energy, TINY)
    best_score = -np.inf
    best_mu = 0.0
    best_selection = np.zeros(N_POINT, dtype=np.int16)
    best_overall: dict[str, float | int] | None = None
    for mu in np.linspace(0.0, 0.20, 2001):
        selection = np.argmax(spectral - mu * normalized_error, axis=0).astype(np.int16)
        point = np.arange(N_POINT)
        overall = aggregate_arrays(
            metrics["pas"][selection, point],
            metrics["pdp"][selection, point],
            metrics["error"][selection, point],
            metrics["energy"],
        )
        if float(overall["score"]) > best_score:
            best_score = float(overall["score"])
            best_mu = float(mu)
            best_selection = selection.copy()
            best_overall = overall
    if best_overall is None:
        raise RuntimeError("point score oracle search produced no result")
    return best_selection, {"mu": best_mu, "metric_only_overall": best_overall}


def oracle_selections(
    names: list[str],
    metrics: dict[str, np.ndarray],
    holes: list[e167.Hole],
    global_to_oof: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    selections = {
        "point_pas": np.argmax(metrics["pas"], axis=0).astype(np.int16),
        "point_nmse": np.argmin(metrics["error"], axis=0).astype(np.int16),
    }
    selections["point_score"], point_score_meta = exact_point_score_selection(metrics)
    hole_selection = np.empty(N_POINT, dtype=np.int16)
    selection_rows: list[dict[str, object]] = []
    for hole in holes:
        rows = global_to_oof[hole.point_ids]
        mask = np.zeros(N_POINT, dtype=bool)
        mask[rows] = True
        scores = []
        for carrier in range(N_CARRIER):
            value = aggregate_arrays(
                metrics["pas"][carrier],
                metrics["pdp"][carrier],
                metrics["error"][carrier],
                metrics["energy"],
                mask,
            )
            scores.append(float(value["score"]))
        best = int(np.argmax(scores))
        hole_selection[rows] = best
        selection_rows.append(
            {
                "oracle": "hole_score",
                "unit": hole.hole_id,
                "carrier_index": best,
                "carrier_name": names[best],
                "hole_score": scores[best],
            }
        )
    selections["hole_score"] = hole_selection
    selection_rows.append(
        {
            "oracle": "point_score",
            "unit": "global",
            "carrier_index": -1,
            "carrier_name": "mixed",
            "hole_score": float(point_score_meta["metric_only_overall"]["score"]),
            "mu": point_score_meta["mu"],
        }
    )
    for oracle, selection in selections.items():
        counts = np.bincount(selection, minlength=N_CARRIER)
        for carrier, count in enumerate(counts):
            if count:
                selection_rows.append(
                    {
                        "oracle": oracle,
                        "unit": "point_count",
                        "carrier_index": carrier,
                        "carrier_name": names[carrier],
                        "count": int(count),
                        "fraction": float(count / N_POINT),
                    }
                )
    return selections, selection_rows


def solve_oracle_weights(force: bool) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if WEIGHT_CACHE.is_file() and not force:
        cache = np.load(WEIGHT_CACHE, allow_pickle=False)
        if cache["convex"].shape == (N_POINT, N_CARRIER) and cache["nonnegative"].shape == (N_POINT, N_CARRIER):
            return cache["convex"], cache["nonnegative"], json.loads(str(cache["summary"]))
    gram_cache = np.load(GRAM_CACHE, allow_pickle=False)
    gram = np.asarray(gram_cache["gram"], dtype=np.complex128).sum(axis=1)
    cross = np.asarray(gram_cache["cross"], dtype=np.complex128).sum(axis=1)
    convex = np.empty((N_POINT, N_CARRIER), dtype=np.float64)
    nonnegative = np.empty((N_POINT, N_CARRIER), dtype=np.float64)
    convex_success = 0
    nonnegative_success = 0
    for point in range(N_POINT):
        q = np.asarray(gram[point].real, dtype=np.float64)
        q = 0.5 * (q + q.T)
        q += np.eye(N_CARRIER) * max(float(np.trace(q) / N_CARRIER), 1e-20) * 1e-12
        b = np.asarray(cross[point].real, dtype=np.float64)
        single_error = np.diag(q) - 2.0 * b
        start = np.zeros(N_CARRIER, dtype=np.float64)
        start[int(np.argmin(single_error))] = 1.0

        def objective(weight: np.ndarray) -> float:
            return float(weight @ q @ weight - 2.0 * b @ weight)

        def gradient(weight: np.ndarray) -> np.ndarray:
            return 2.0 * (q @ weight - b)

        result = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * N_CARRIER,
            constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones_like(w)},
            options={"ftol": 1e-11, "maxiter": 500, "disp": False},
        )
        weight = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
        weight /= max(float(np.sum(weight)), TINY)
        convex[point] = weight
        convex_success += int(bool(result.success))

        result_nn = minimize(
            objective,
            start,
            jac=gradient,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * N_CARRIER,
            options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
        )
        nonnegative[point] = np.maximum(np.asarray(result_nn.x, dtype=np.float64), 0.0)
        nonnegative_success += int(bool(result_nn.success))
        if (point + 1) % 50 == 0 or point + 1 == N_POINT:
            print(f"E173 convex QP {point + 1}/{N_POINT}", flush=True)
    summary = {
        "convex_success": convex_success,
        "nonnegative_success": nonnegative_success,
        "convex_effective_count_mean": float(np.mean(1.0 / np.maximum(np.sum(convex * convex, axis=1), TINY))),
        "convex_max_weight_mean": float(np.mean(np.max(convex, axis=1))),
        "nonnegative_sum_mean": float(np.mean(np.sum(nonnegative, axis=1))),
        "nonnegative_active_mean": float(np.mean(np.sum(nonnegative > 1e-4, axis=1))),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        WEIGHT_CACHE,
        convex=convex.astype(np.float32),
        nonnegative=nonnegative.astype(np.float32),
        summary=np.asarray(json.dumps(summary)),
    )
    return convex, nonnegative, summary


class SourceProvider:
    def __init__(
        self,
        bank: np.ndarray,
        selection: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        array: np.ndarray | None = None,
    ) -> None:
        self.bank = bank
        self.selection = selection
        self.weights = weights
        self.array = array

    def block(self, start: int, end: int) -> np.ndarray:
        if self.array is not None:
            return np.array(self.array[start:end], dtype=np.complex64, copy=True)
        candidate = np.asarray(self.bank[:, start:end], dtype=np.complex64)
        if self.selection is not None:
            local = self.selection[start:end]
            return np.asarray(candidate[local, np.arange(end - start)], dtype=np.complex64)
        if self.weights is not None:
            return np.einsum(
                "cbmns,bc->bmns",
                candidate,
                np.asarray(self.weights[start:end], dtype=np.float32),
                optimize=True,
            ).astype(np.complex64)
        raise RuntimeError("source provider has no selection, weights, or array")


def score_provider(
    provider: SourceProvider,
    device: torch.device,
    batch_size: int,
    query_ids: np.ndarray,
) -> dict[str, object]:
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    pas = np.empty(N_POINT, dtype=np.float64)
    pdp = np.empty(N_POINT, dtype=np.float64)
    error = np.empty(N_POINT, dtype=np.float64)
    energy = np.empty(N_POINT, dtype=np.float64)
    for start in range(0, N_POINT, batch_size):
        end = min(start + batch_size, N_POINT)
        prediction = torch.as_tensor(provider.block(start, end), device=device)
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            values = point_metrics_torch(prediction, truth)
        pas[start:end], pdp[start:end], error[start:end], energy[start:end] = values[:4]
    return {
        "pas": pas,
        "pdp": pdp,
        "error": error,
        "energy": energy,
        "overall": aggregate_arrays(pas, pdp, error, energy),
    }


def projection_context() -> dict[str, object]:
    selection = json.loads(V4_SELECTION.read_text(encoding="utf-8"))
    policy = {int(row["site"]): str(row["candidate"]) for row in selection["final_policy"]}
    lifecycle = json.loads(LIFECYCLE_SELECTION.read_text(encoding="utf-8"))["winner"]
    decisions = {
        (int(row["held_fold"]), int(row["site"]), REGIME_NAMES.index(str(row["regime"]))): str(row["candidate"])
        for row in lifecycle["nested_decisions"]
    }
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    return {
        "query_ids": query_ids,
        "fold_id": fold_id,
        "site": site,
        "regime": regime,
        "policy": policy,
        "lifecycle_decisions": decisions,
        "ad": np.load(TARGET_AD, mmap_mode="r"),
        "pas": np.load(TARGET_PAS, mmap_mode="r"),
        "pdp": np.load(TARGET_PDP, mmap_mode="r"),
        "calibrated_pas": np.load(OOF_CALIBRATED_PAS, mmap_mode="r"),
        "calibrated_pdp": np.load(OOF_CALIBRATED_PDP, mmap_mode="r"),
        "lifecycle_target": np.load(TARGET_PATHS["lifecycle_shift"][0], mmap_mode="r"),
    }


def project_block(
    source: np.ndarray,
    start: int,
    end: int,
    context: dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    ad, raw_pas, raw_pdp = target_batch(
        context["ad"], context["pas"], context["pdp"], start, end, device
    )
    calibrated_pas = torch.as_tensor(
        np.asarray(context["calibrated_pas"][start:end], dtype=np.float32), device=device
    )
    calibrated_pdp = torch.as_tensor(
        np.asarray(context["calibrated_pdp"][start:end], dtype=np.float32), device=device
    )
    base = torch.as_tensor(source, device=device)
    output = torch.empty_like(base)
    local_site = context["site"][start:end]
    with torch.inference_mode():
        for site_value in np.unique(local_site):
            rows = np.flatnonzero(local_site == site_value)
            output[rows] = _parse_v4_candidate(
                base[rows],
                ad[rows],
                raw_pas[rows],
                raw_pdp[rows],
                calibrated_pas[rows],
                calibrated_pdp[rows],
                context["policy"][int(site_value)],
            )
    return output


def support_block(channel: torch.Tensor) -> torch.Tensor:
    shaped = channel.reshape(len(channel), N_POL, N_H, N_V, N_UE, N_SC)
    transformed = torch.fft.ifft(torch.fft.fft2(shaped, dim=(2, 3)), dim=-1)
    power = transformed.abs().square().sum(dim=(1, 4)).float()
    power /= power.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
    return power.to(torch.float16).float()


def postprocess_provider(
    provider: SourceProvider,
    context: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    query_ids = context["query_ids"]
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    predicted_energy = np.empty(N_POINT, dtype=np.float64)
    truth_energy = np.empty(N_POINT, dtype=np.float64)
    cross = np.empty(N_POINT, dtype=np.float64)
    for start in range(0, N_POINT, batch_size):
        end = min(start + batch_size, N_POINT)
        projected = project_block(provider.block(start, end), start, end, context, device)
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            predicted_energy[start:end] = projected.abs().square().sum(
                dim=(1, 2, 3), dtype=torch.float64
            ).cpu().numpy()
            truth_energy[start:end] = truth.abs().square().sum(
                dim=(1, 2, 3), dtype=torch.float64
            ).cpu().numpy()
            cross[start:end] = (projected * truth.conj()).sum(
                dim=(1, 2, 3), dtype=torch.complex128
            ).real.cpu().numpy()
    scale, decisions = fit_scales(
        predicted_energy,
        truth_energy,
        cross,
        context["site"],
        context["fold_id"],
        nested=True,
    )

    pas = np.empty(N_POINT, dtype=np.float64)
    pdp = np.empty(N_POINT, dtype=np.float64)
    error = np.empty(N_POINT, dtype=np.float64)
    energy = np.empty(N_POINT, dtype=np.float64)
    for start in range(0, N_POINT, batch_size):
        end = min(start + batch_size, N_POINT)
        projected = project_block(provider.block(start, end), start, end, context, device)
        projected *= torch.as_tensor(scale[start:end, None, None, None], device=device)
        current = support_block(projected)
        output = projected.clone()
        local_fold = context["fold_id"][start:end]
        local_site = context["site"][start:end]
        local_regime = context["regime"][start:end]
        for fold_value in np.unique(local_fold):
            for site_value in np.unique(local_site):
                for regime_value in np.unique(local_regime):
                    rows = np.flatnonzero(
                        (local_fold == fold_value)
                        & (local_site == site_value)
                        & (local_regime == regime_value)
                    )
                    if len(rows) == 0:
                        continue
                    name = context["lifecycle_decisions"][
                        (int(fold_value), int(site_value), int(regime_value))
                    ]
                    if name == "identity":
                        continue
                    candidate = _candidate_from_name(name)
                    with torch.inference_mode():
                        output[rows] = apply_candidate(
                            output[rows].clone(),
                            current[rows],
                            torch.as_tensor(
                                np.asarray(context["lifecycle_target"][start:end][rows], dtype=np.float32),
                                device=device,
                            ),
                            local_regime[rows],
                            local_site[rows],
                            candidate,
                        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            values = point_metrics_torch(output, truth)
        pas[start:end], pdp[start:end], error[start:end], energy[start:end] = values[:4]
        print(f"E173 V7 postprocess {end}/{N_POINT}", flush=True)
    return {
        "pas": pas,
        "pdp": pdp,
        "error": error,
        "energy": energy,
        "overall": aggregate_arrays(pas, pdp, error, energy),
        "scale_decisions": decisions,
    }


def build_oracle_hole_rows(
    oracle_results: dict[str, dict[str, object]],
    holes: list[e167.Hole],
    global_to_oof: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for oracle, result in oracle_results.items():
        for hole in holes:
            point_rows = global_to_oof[hole.point_ids]
            mask = np.zeros(N_POINT, dtype=bool)
            mask[point_rows] = True
            rows.append(
                {
                    "oracle": oracle,
                    "hole_id": hole.hole_id,
                    "site": hole.site,
                    "fold": hole.ordinal,
                    **aggregate_arrays(
                        result["pas"], result["pdp"], result["error"], result["energy"], mask
                    ),
                }
            )
    return rows


def v12_candidate_audit(
    metrics: dict[str, np.ndarray],
    baselines: dict[str, dict[str, object]],
    residual: np.ndarray,
) -> list[dict[str, object]]:
    v7 = baselines["V7_complete"]
    v12 = baselines["V12_offline_reference"]
    best_pas = np.max(metrics["pas"], axis=0)
    v7_compact = np.asarray(np.load(V7_COMPACT, mmap_mode="r"), dtype=np.float32)
    v7_root_residual = np.sqrt(np.maximum(v7_compact, 0.0)) - np.sqrt(
        np.maximum(np.asarray(np.load(TRAIN_COMPACT, mmap_mode="r"))[np.load(OOF_META)["query_ids"]], 0.0)
    )
    v7_norm = np.linalg.norm(v7_root_residual.reshape(N_POINT, -1), axis=1)
    carrier_norm = np.linalg.norm(residual.reshape(N_CARRIER, N_POINT, -1), axis=2)
    best_norm = np.min(carrier_norm, axis=0)
    rows: list[dict[str, object]] = []
    for label, mask in (
        ("V12_PAS_positive", np.asarray(v12["pas"]) > np.asarray(v7["pas"])),
        ("V12_PAS_gain_gt_0.05", np.asarray(v12["pas"]) - np.asarray(v7["pas"]) > 0.05),
        ("V12_total_point_proxy_positive", 0.4 * (np.asarray(v12["pas"]) - np.asarray(v7["pas"])) + 0.4 * (np.asarray(v12["pdp"]) - np.asarray(v7["pdp"])) > 0.0),
    ):
        rows.append(
            {
                "subset": label,
                "count": int(np.sum(mask)),
                "best_carrier_pas_gt_v7_fraction": float(np.mean(best_pas[mask] > np.asarray(v7["pas"])[mask])) if np.any(mask) else float("nan"),
                "best_carrier_pas_ge_v12_fraction": float(np.mean(best_pas[mask] >= np.asarray(v12["pas"])[mask])) if np.any(mask) else float("nan"),
                "best_carrier_compact_residual_lt_v7_fraction": float(np.mean(best_norm[mask] < v7_norm[mask])) if np.any(mask) else float("nan"),
                "mean_best_carrier_pas_minus_v7": float(np.mean(best_pas[mask] - np.asarray(v7["pas"])[mask])) if np.any(mask) else float("nan"),
                "mean_v12_pas_minus_v7": float(np.mean(np.asarray(v12["pas"])[mask] - np.asarray(v7["pas"])[mask])) if np.any(mask) else float("nan"),
            }
        )
    return rows


def fusion_selection_audit(
    metrics: dict[str, np.ndarray],
    baselines: dict[str, dict[str, object]],
    holes: list[e167.Hole],
    global_to_oof: np.ndarray,
    point_score_mu: float,
) -> list[dict[str, object]]:
    """Measure how often a raw carrier already beats the fixed fusion at a point."""
    total_energy = float(np.sum(metrics["energy"]))
    utility_scale = point_score_mu * N_POINT / max(total_energy, TINY)
    carrier_utility = 0.4 * (metrics["pas"] + metrics["pdp"]) - utility_scale * metrics["error"]
    best_pas = np.max(metrics["pas"], axis=0)
    best_pdp = np.max(metrics["pdp"], axis=0)
    best_utility = np.max(carrier_utility, axis=0)

    scopes: list[tuple[str, np.ndarray]] = [("fixed500", np.ones(N_POINT, dtype=bool))]
    for hole in holes:
        mask = np.zeros(N_POINT, dtype=bool)
        mask[global_to_oof[hole.point_ids]] = True
        scopes.append((hole.hole_id, mask))

    rows: list[dict[str, object]] = []
    for scope, mask in scopes:
        row: dict[str, object] = {"scope": scope, "count": int(np.sum(mask))}
        for baseline_name in ("raw_ridge_stack", "V7_complete"):
            baseline = baselines[baseline_name]
            baseline_utility = (
                0.4 * (np.asarray(baseline["pas"]) + np.asarray(baseline["pdp"]))
                - utility_scale * np.asarray(baseline["error"])
            )
            pas_gain = best_pas - np.asarray(baseline["pas"])
            pdp_gain = best_pdp - np.asarray(baseline["pdp"])
            utility_gain = best_utility - baseline_utility
            prefix = "vs_raw_ridge" if baseline_name == "raw_ridge_stack" else "vs_v7"
            for metric_name, gain in (
                ("pas", pas_gain),
                ("pdp", pdp_gain),
                ("point_score_utility", utility_gain),
            ):
                positive = gain[mask] > 1e-12
                row[f"{prefix}_{metric_name}_better_count"] = int(np.sum(positive))
                row[f"{prefix}_{metric_name}_better_fraction"] = float(np.mean(positive))
                row[f"{prefix}_{metric_name}_mean_best_minus_baseline"] = float(
                    np.mean(gain[mask])
                )
        rows.append(row)
    return rows


def make_plots(
    names: list[str],
    overall_rows: list[dict[str, object]],
    correlation: dict[str, object],
    hard_rows: list[dict[str, object]],
    selections: dict[str, np.ndarray],
) -> list[Path]:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    carrier_rows = [row for row in overall_rows if row["kind"] == "carrier"]
    order = np.argsort([float(row["score"]) for row in carrier_rows])[::-1]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(N_CARRIER), [float(carrier_rows[index]["score"]) for index in order])
    ax.axhline(EXPECTED_V7["score"], color="red", linestyle="--", label="V7 complete")
    ax.set_xticks(np.arange(N_CARRIER), [names[index] for index in order], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("official score")
    ax.set_title("17 Clean carrier fixed500 scores")
    ax.legend()
    fig.tight_layout()
    path = VIS_DIR / "01_十七载体完整官方总分.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(correlation["pearson_matrix"], vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(N_CARRIER), names, rotation=90, fontsize=7)
    ax.set_yticks(range(N_CARRIER), names, fontsize=7)
    ax.set_title("Compact PAS root-residual Pearson correlation")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path = VIS_DIR / "02_载体PAS残差相关矩阵.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    hole_ids = [row["hole_id"] for row in hard_rows]
    ax.bar(np.arange(len(hard_rows)) - 0.18, [row["best_single_score"] for row in hard_rows], width=0.36, label="best raw carrier")
    ax.bar(np.arange(len(hard_rows)) + 0.18, [row["v7_score"] for row in hard_rows], width=0.36, label="V7")
    ax.set_xticks(np.arange(len(hard_rows)), hole_ids)
    ax.set_ylabel("official score")
    ax.set_title("Hard holes: best single carrier vs V7")
    ax.legend()
    fig.tight_layout()
    path = VIS_DIR / "03_困难洞最佳单载体与V7.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    selection = selections["point_score"]
    counts = np.bincount(selection, minlength=N_CARRIER)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(N_CARRIER), counts)
    ax.set_xticks(np.arange(N_CARRIER), names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("selected points")
    ax.set_title("Point-score GT oracle carrier choices")
    fig.tight_layout()
    path = VIS_DIR / "04_逐点总分Oracle选择频数.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def write_report(report: dict[str, object]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(json_compatible(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    decision = report["decision"]
    best_raw = report["best_raw_carrier"]
    best_post = report["best_postprocessed_oracle"]
    corr = report["residual_correlation_summary"]
    v12 = report["v12_candidate_audit"][0]
    fusion = report["fusion_selection_audit"][0]
    raw_oracles = report["raw_oracles"]
    text = f"""# V7十七路Clean答案到底有没有正确候选

## 一句话结论

**{decision['plain_conclusion']}**

最强单一原始carrier是`{best_raw['name']}`，完整500点PAS/PDP/NMSE/总分=`{best_raw['pas']:.6f}/{best_raw['pdp']:.6f}/{best_raw['nmse']:.6f}/{best_raw['score']:.6f}`。完整V7为`0.597983/0.756446/0.947710/0.644456`。

最强GT oracle经过与V7相同的冻结投影、严格尺度和生命周期后是`{best_post['oracle']}`，PAS/PDP/NMSE/总分=`{best_post['pas']:.6f}/{best_post['pdp']:.6f}/{best_post['nmse']:.6f}/{best_post['score']:.6f}`，相对V7=`{best_post['delta_vs_v7']:+.6f}`。

## 正确carrier是否经常被融合丢掉

在全部500点中，至少一个原始carrier的PAS高于原始ridge融合的比例=`{fusion['vs_raw_ridge_pas_better_fraction']:.1%}`，高于完整V7的比例=`{fusion['vs_v7_pas_better_fraction']:.1%}`。按`point_score` oracle实际使用的逐点选择效用计算，至少一个carrier优于原始ridge/完整V7的比例分别为`{fusion['vs_raw_ridge_point_score_utility_better_fraction']:.1%}/{fusion['vs_v7_point_score_utility_better_fraction']:.1%}`。

但“经常存在某个单项更好的carrier”不等于这些carrier能拼出大幅更高的完整信道。原始逐点`point_score` oracle总分=`{raw_oracles['point_score']['score']:.6f}`，相对原始ridge=`{raw_oracles['point_score']['score'] - 0.5719807434967276:+.6f}`；经过冻结V7后处理后只剩`{best_post['score']:.6f}`，相对完整V7=`{best_post['delta_vs_v7']:+.6f}`。

## 载体是否真的多样

17路compact PAS平方根残差两两Pearson均值/中位数=`{corr['compact_root_residual_pearson_mean']:.3f}/{corr['compact_root_residual_pearson_median']:.3f}`，最小/最大=`{corr['compact_root_residual_pearson_min']:.3f}/{corr['compact_root_residual_pearson_max']:.3f}`。相关性越接近1，说明载体犯的是同一种PAS错误；明显低于1才代表可选择的互补答案。

## V12修对点里是否已有Clean候选

在V12相对V7 PAS正向的`{v12['count']}`个点中，至少一个Clean carrier PAS高于V7的比例=`{v12['best_carrier_pas_gt_v7_fraction']:.1%}`，达到或超过V12 PAS的比例=`{v12['best_carrier_pas_ge_v12_fraction']:.1%}`，compact PAS残差优于V7的比例=`{v12['best_carrier_compact_residual_lt_v7_fraction']:.1%}`。

## 裁决口径

最强完整V7流程oracle相对V7增益低于`+0.005`视为候选本身不足；`+0.005~+0.015`为有限选择余量；至少`+0.015`才说明正确答案经常已经存在、主要问题是融合选择。

本轮全部carrier与oracle只使用官方Clean数据。V12只用于最后一段离线错误对照，没有进入任何选择、权重或投影。本轮没有训练gate、网络或生成提交版本。
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    names = load_names()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    metrics = build_carrier_metrics(device, args.batch_size, args.force_metrics)
    gram = np.load(GRAM_CACHE, allow_pickle=False)
    gram_point_error = (
        np.asarray(gram["gram"][:, :, np.arange(N_CARRIER), np.arange(N_CARRIER)].real).sum(axis=1).T
        - 2.0 * np.asarray(gram["cross"][:, :, :,].real).sum(axis=1).T
        + np.asarray(gram["energy"]).sum(axis=1)[None]
    )
    metric_error = np.asarray(metrics["error"])
    gram_absolute_error = np.abs(gram_point_error - metric_error)
    gram_identity = float(np.max(gram_absolute_error / np.maximum(metric_error, 1e-20)))
    gram_max_absolute_error = float(np.max(gram_absolute_error))
    point_nmse_oracle = float(np.min(metric_error, axis=0).sum() / np.sum(metrics["energy"]))
    if (
        abs(point_nmse_oracle - EXPECTED_POINT_NMSE_ORACLE) > 1e-6
        or gram_max_absolute_error > 2e-8
    ):
        raise RuntimeError(
            "E45 identity gate failed: "
            f"oracle={point_nmse_oracle}, gram_rel={gram_identity}, "
            f"gram_abs={gram_max_absolute_error}"
        )

    baselines = baseline_metrics(device, args.batch_size, query_ids)
    holes, global_to_oof = canonical_holes()
    overall_rows, hole_rows = carrier_tables(names, metrics, baselines, holes, global_to_oof)
    pair_rows, correlation = residual_correlations(names, metrics, query_ids)
    selections, selection_rows = oracle_selections(names, metrics, holes, global_to_oof)
    point_score_mu = float(
        next(
            row["mu"]
            for row in selection_rows
            if row["oracle"] == "point_score" and row["unit"] == "global"
        )
    )
    convex, nonnegative, weight_summary = solve_oracle_weights(args.force_weights)
    bank = np.load(CARRIER_BANK, mmap_mode="r")
    providers = {
        name: SourceProvider(bank, selection=selection) for name, selection in selections.items()
    }
    providers["convex_nmse"] = SourceProvider(bank, weights=convex)
    providers["nonnegative_nmse"] = SourceProvider(bank, weights=nonnegative)

    raw_oracles: dict[str, dict[str, object]] = {}
    oracle_rows: list[dict[str, object]] = []
    for name, provider in providers.items():
        result = score_provider(provider, device, args.batch_size, query_ids)
        raw_oracles[name] = result
        oracle_rows.append({"stage": "raw", "oracle": name, **result["overall"]})
        print(f"E173 raw oracle {name}: {result['overall']['score']:.6f}", flush=True)

    point = np.arange(N_POINT)
    for name in ("point_pas", "point_nmse", "point_score", "hole_score"):
        selected = selections[name]
        selected_stats = aggregate_arrays(
            metrics["pas"][selected, point],
            metrics["pdp"][selected, point],
            metrics["error"][selected, point],
            metrics["energy"],
        )
        if max(abs(float(selected_stats[key]) - float(raw_oracles[name]["overall"][key])) for key in ("pas", "pdp", "nmse", "score")) > 5e-6:
            raise RuntimeError(f"oracle reconstruction identity failed: {name}")

    context = projection_context()
    stack_post = postprocess_provider(
        SourceProvider(bank, array=np.load(RAW_STACK, mmap_mode="r")),
        context,
        device,
        args.batch_size,
    )
    if max(abs(float(stack_post["overall"][key]) - EXPECTED_V7[key]) for key in EXPECTED_V7) > 5e-6:
        raise RuntimeError(f"V7 postprocess identity failed: {stack_post['overall']}")

    postprocessed: dict[str, dict[str, object]] = {}
    for name in ("point_pas", "point_score", "hole_score", "convex_nmse"):
        result = postprocess_provider(providers[name], context, device, args.batch_size)
        postprocessed[name] = result
        oracle_rows.append({"stage": "v7_frozen_postprocess", "oracle": name, **result["overall"]})
        print(f"E173 postprocessed oracle {name}: {result['overall']['score']:.6f}", flush=True)

    oracle_results_for_holes = {f"raw_{name}": result for name, result in raw_oracles.items()}
    oracle_results_for_holes.update({f"post_{name}": result for name, result in postprocessed.items()})
    oracle_hole_rows = build_oracle_hole_rows(
        oracle_results_for_holes, holes, global_to_oof
    )

    hard_rows: list[dict[str, object]] = []
    v7_hole_lookup = {
        row["hole_id"]: row
        for row in hole_rows
        if row["kind"] == "baseline" and row["name"] == "V7_complete"
    }
    carrier_hole_lookup = {}
    for row in hole_rows:
        if row["kind"] == "carrier":
            carrier_hole_lookup.setdefault(row["hole_id"], []).append(row)
    for hole_id in HARD_HOLES:
        best = max(carrier_hole_lookup[hole_id], key=lambda row: float(row["score"]))
        v7 = v7_hole_lookup[hole_id]
        post_hole = next(
            row for row in oracle_hole_rows if row["oracle"] == "post_hole_score" and row["hole_id"] == hole_id
        )
        hard_rows.append(
            {
                "hole_id": hole_id,
                "best_single_carrier": best["name"],
                "best_single_pas": best["pas"],
                "best_single_pdp": best["pdp"],
                "best_single_nmse": best["nmse"],
                "best_single_score": best["score"],
                "v7_pas": v7["pas"],
                "v7_pdp": v7["pdp"],
                "v7_nmse": v7["nmse"],
                "v7_score": v7["score"],
                "best_single_minus_v7": float(best["score"]) - float(v7["score"]),
                "post_hole_oracle_score": post_hole["score"],
                "post_hole_oracle_minus_v7": float(post_hole["score"]) - float(v7["score"]),
            }
        )

    v12_rows = v12_candidate_audit(metrics, baselines, correlation["compact_residual"])
    fusion_rows = fusion_selection_audit(
        metrics, baselines, holes, global_to_oof, point_score_mu
    )
    weight_rows = []
    for name, weight in (("convex_nmse", convex), ("nonnegative_nmse", nonnegative)):
        normalized = weight / np.maximum(np.sum(weight, axis=1, keepdims=True), TINY)
        for carrier, carrier_name in enumerate(names):
            weight_rows.append(
                {
                    "oracle": name,
                    "carrier_index": carrier,
                    "carrier_name": carrier_name,
                    "mean_weight": float(np.mean(weight[:, carrier])),
                    "mean_normalized_weight": float(np.mean(normalized[:, carrier])),
                    "active_fraction": float(np.mean(weight[:, carrier] > 1e-4)),
                }
            )

    best_raw = max(
        (row for row in overall_rows if row["kind"] == "carrier"),
        key=lambda row: float(row["score"]),
    )
    best_post_name, best_post_value = max(
        postprocessed.items(), key=lambda item: float(item[1]["overall"]["score"])
    )
    best_post = {
        "oracle": best_post_name,
        **best_post_value["overall"],
        "delta_vs_v7": float(best_post_value["overall"]["score"]) - EXPECTED_V7["score"],
    }
    delta = float(best_post["delta_vs_v7"])
    if delta < 0.005:
        plain = "17个Clean载体即使由GT挑选并经过完整V7流程，也只比V7高一点；主要瓶颈是候选本身缺少正确传播状态，不是融合选择。"
        level = "carrier_content_limited"
    elif delta < 0.015:
        plain = "17个Clean载体中存在有限的选择余量，但不足以解释大幅提分；候选内容和融合选择同时受限。"
        level = "limited_selection_headroom"
    else:
        plain = "正确答案经常已经存在于17个Clean载体中，V7的主要损失来自固定融合没有按点或按洞选对。"
        level = "large_selection_headroom"

    overall_rows.extend(
        {"kind": "oracle", "name": f"{row['stage']}:{row['oracle']}", **{key: row[key] for key in ("count", "pas", "pdp", "nmse", "nmse_score", "score")}}
        for row in oracle_rows
    )
    plot_paths = make_plots(names, overall_rows, correlation, hard_rows, selections)
    write_csv(OVERALL_CSV, overall_rows)
    write_csv(HOLE_CSV, hole_rows)
    write_csv(PAIR_CSV, pair_rows)
    write_csv(ORACLE_CSV, oracle_rows)
    write_csv(ORACLE_HOLE_CSV, oracle_hole_rows)
    write_csv(SELECTION_CSV, selection_rows)
    write_csv(HARD_HOLE_CSV, hard_rows)
    write_csv(V12_CSV, v12_rows)
    write_csv(WEIGHT_CSV, weight_rows)
    write_csv(FUSION_CSV, fusion_rows)

    report = {
        "version": VERSION,
        "status": "analysis_complete",
        "clean_lineage": {
            "carrier_official_inputs_only": True,
            "sionna_used_in_carrier_or_oracle": False,
            "v12_role": "offline point-subset comparison only",
        },
        "identity_gates": {
            "e45_point_oracle_nmse": point_nmse_oracle,
            "expected_e45_point_oracle_nmse": EXPECTED_POINT_NMSE_ORACLE,
            "gram_metric_max_relative_error": gram_identity,
            "gram_metric_max_absolute_error": gram_max_absolute_error,
            "v7_postprocess_replay": stack_post["overall"],
        },
        "carrier_names": names,
        "best_raw_carrier": best_raw,
        "residual_correlation_summary": {
            key: value for key, value in correlation.items() if not isinstance(value, np.ndarray)
        },
        "raw_oracles": {name: value["overall"] for name, value in raw_oracles.items()},
        "postprocessed_oracles": {name: value["overall"] for name, value in postprocessed.items()},
        "best_postprocessed_oracle": best_post,
        "weight_summary": weight_summary,
        "hard_holes": hard_rows,
        "v12_candidate_audit": v12_rows,
        "fusion_selection_audit": fusion_rows,
        "decision": {
            "level": level,
            "plain_conclusion": plain,
            "threshold_small": 0.005,
            "threshold_large": 0.015,
        },
        "outputs": [
            PATHS.relative(path)
            for path in (
                REPORT_JSON,
                REPORT_MD,
                OVERALL_CSV,
                HOLE_CSV,
                PAIR_CSV,
                ORACLE_CSV,
                ORACLE_HOLE_CSV,
                SELECTION_CSV,
                HARD_HOLE_CSV,
                V12_CSV,
                WEIGHT_CSV,
                FUSION_CSV,
                *plot_paths,
            )
        ],
        "elapsed_seconds": time.time() - started,
    }
    write_report(report)
    print(json.dumps(json_compatible(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
