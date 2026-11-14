#!/usr/bin/env python3
"""Audit the low-dimensional geometry of E180's 13 Clean channel actions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import resource
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import analyze_e180_clean_point_action_oracle as e180
from common.official_metrics import OfficialMetricAccumulator, aggregate_statistics
from project_paths import PATHS


VERSION = "Round2-E182-clean-action-low-rank"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e182_clean_action_low_rank"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e182_clean_action_low_rank"

BASIS_H = ARTIFACT_DIR / "scaled_complex_h_directions_top4.npy"
ORACLE_H_PATTERN = "bounded_continuous_oracle_dim{dimension}.npy"
COEFFICIENTS = ARTIFACT_DIR / "bounded_continuous_oracle_coefficients.npz"
GEOMETRY_CACHE = ARTIFACT_DIR / "action_difference_geometry.npz"
REPORT_JSON = ANALYSIS_DIR / "clean_action_low_rank.json"
REPORT_MD = ANALYSIS_DIR / "Clean动作是离散选择还是少数连续方向.md"
DIMENSION_CSV = ANALYSIS_DIR / "动作差分维数.csv"
PAIR_CSV = ANALYSIS_DIR / "动作差分两两相似度.csv"
COORDINATE_CSV = ANALYSIS_DIR / "动作低维坐标.csv"
ORACLE_CSV = ANALYSIS_DIR / "低维连续动作oracle官方分.csv"

N_POINT = 500
N_ACTION = 13
N_DELTA = N_ACTION - 1
N_DIRECTION = 4
EXPECTED_SHAPE = (N_POINT, 256, 4, 192)
EXPECTED_V7_SCORE = 0.6444563457343335
EXPECTED_E180_SCORE = 0.677201175970
EXPECTED_EXPANDED_SCORE = 0.693770154
SUCCESS_SCORE_2D = 0.660000
SUCCESS_SCORE_4D = EXPECTED_E180_SCORE
SUCCESS_EXPLAINED_4D = 0.90
TINY = 1e-30

REPRESENTATIONS = (
    "delta_h",
    "delta_pas_power",
    "delta_pdp_power",
    "delta_pas_l2_shape",
    "delta_pdp_l2_shape",
    "delta_pas_sqrt_shape",
    "delta_pdp_sqrt_shape",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--geometry-batch-size", type=int, default=4)
    parser.add_argument("--opt-batch-size", type=int, default=100)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--force-basis", action="store_true")
    parser.add_argument("--force-oracle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def load_clean_actions() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[e180.ActionSpec],
    e180.ActionProvider,
]:
    metadata = np.load(e180.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    e168 = np.load(e180.E168_METRICS, allow_pickle=False)
    site = e168["site"].astype(np.int8)
    if not np.array_equal(query_ids, e168["query_ids"].astype(np.int64)):
        raise RuntimeError("E168 canonical query order mismatch")
    all_actions = e180.build_actions(fold_id, site)
    actions = [action for action in all_actions if action.primary]
    if len(actions) != N_ACTION or actions[0].action_id != "V7":
        raise RuntimeError(f"unexpected E180 primary action order: {[a.action_id for a in actions]}")
    provider = e180.ActionProvider(actions)
    for action in actions:
        block = provider.block(action.action_id, 0, 1)
        if block.shape != EXPECTED_SHAPE[1:] and block.shape != (1, *EXPECTED_SHAPE[1:]):
            raise ValueError(f"invalid action block {action.action_id}: {block.shape}")
        if block.shape == (1, *EXPECTED_SHAPE[1:]) and (
            block.dtype != np.complex64 or not np.isfinite(block).all()
        ):
            raise ValueError(f"invalid action data {action.action_id}: {block.dtype}")
    return query_ids, fold_id, site, actions, provider


def update_real_geometry(
    state: dict[str, dict[str, np.ndarray | float | int]],
    name: str,
    values: torch.Tensor,
) -> None:
    flat = values.reshape(N_DELTA, -1).float()
    gram = (flat @ flat.T).double().cpu().numpy()
    sums = flat.sum(dim=1, dtype=torch.float64).cpu().numpy()
    entry = state[name]
    entry["gram"] = np.asarray(entry["gram"]) + gram
    entry["sum"] = np.asarray(entry["sum"]) + sums
    entry["count"] = int(entry["count"]) + flat.shape[1]


def update_complex_geometry(
    state: dict[str, dict[str, np.ndarray | float | int]],
    name: str,
    values: torch.Tensor,
) -> None:
    flat = values.reshape(N_DELTA, -1)
    gram = torch.real(flat.conj() @ flat.T).double().cpu().numpy()
    sums = (flat.real.sum(dim=1, dtype=torch.float64) + flat.imag.sum(dim=1, dtype=torch.float64)).cpu().numpy()
    entry = state[name]
    entry["gram"] = np.asarray(entry["gram"]) + gram
    entry["sum"] = np.asarray(entry["sum"]) + sums
    entry["count"] = int(entry["count"]) + 2 * flat.shape[1]


def l2_normalize(values: torch.Tensor, dim: int) -> torch.Tensor:
    denominator = values.square().sum(dim=dim, keepdim=True).sqrt()
    return torch.where(denominator > 0.0, values / denominator.clamp_min(1e-20), torch.zeros_like(values))


def action_spectra(channels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = channels.shape[0]
    shaped = channels.reshape(count, 2, 16, 8, 4, 192)
    pas = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
    pas = pas.reshape(count, 256, 4, 192)
    pdp = torch.fft.ifft(channels, dim=-1).abs().square().float()
    return pas, pdp


def accumulate_geometry(
    actions: list[e180.ActionSpec],
    provider: e180.ActionProvider,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> dict[str, np.ndarray]:
    required = {
        "representation_names",
        "grams",
        "sums",
        "counts",
        "point_action_norm_sq",
    }
    if GEOMETRY_CACHE.is_file() and not force:
        cache = np.load(GEOMETRY_CACHE, allow_pickle=False)
        if required.issubset(cache.files) and cache["representation_names"].astype(str).tolist() == list(REPRESENTATIONS):
            return {key: cache[key] for key in cache.files}

    state: dict[str, dict[str, np.ndarray | float | int]] = {
        name: {
            "gram": np.zeros((N_DELTA, N_DELTA), dtype=np.float64),
            "sum": np.zeros(N_DELTA, dtype=np.float64),
            "count": 0,
        }
        for name in REPRESENTATIONS
    }
    point_action_norm_sq = np.empty((N_DELTA, N_POINT), dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, N_POINT, batch_size):
            end = min(start + batch_size, N_POINT)
            baseline = torch.as_tensor(provider.block("V7", start, end), device=device)
            action_blocks = [baseline]
            for action in actions[1:]:
                action_blocks.append(torch.as_tensor(provider.block(action.action_id, start, end), device=device))
            channels = torch.stack(action_blocks, dim=0)
            delta_h = channels[1:] - channels[:1]
            point_action_norm_sq[:, start:end] = (
                delta_h.abs().square().sum(dim=(2, 3, 4), dtype=torch.float64).cpu().numpy()
            )
            update_complex_geometry(state, "delta_h", delta_h)

            flat_channels = channels.reshape(N_ACTION * (end - start), 256, 4, 192)
            pas, pdp = action_spectra(flat_channels)
            pas = pas.reshape(N_ACTION, end - start, 256, 4, 192)
            pdp = pdp.reshape(N_ACTION, end - start, 256, 4, 192)
            delta_pas = pas[1:] - pas[:1]
            delta_pdp = pdp[1:] - pdp[:1]
            update_real_geometry(state, "delta_pas_power", delta_pas)
            update_real_geometry(state, "delta_pdp_power", delta_pdp)

            pas_shape = l2_normalize(pas, dim=2)
            pdp_shape = l2_normalize(pdp, dim=4)
            update_real_geometry(state, "delta_pas_l2_shape", pas_shape[1:] - pas_shape[:1])
            update_real_geometry(state, "delta_pdp_l2_shape", pdp_shape[1:] - pdp_shape[:1])

            pas_sqrt = l2_normalize(pas.clamp_min(0.0).sqrt(), dim=2)
            pdp_sqrt = l2_normalize(pdp.clamp_min(0.0).sqrt(), dim=4)
            update_real_geometry(state, "delta_pas_sqrt_shape", pas_sqrt[1:] - pas_sqrt[:1])
            update_real_geometry(state, "delta_pdp_sqrt_shape", pdp_sqrt[1:] - pdp_sqrt[:1])
            print(f"E182 geometry points {end}/{N_POINT}", flush=True)

    payload = {
        "representation_names": np.asarray(REPRESENTATIONS),
        "grams": np.stack([np.asarray(state[name]["gram"]) for name in REPRESENTATIONS]),
        "sums": np.stack([np.asarray(state[name]["sum"]) for name in REPRESENTATIONS]),
        "counts": np.asarray([int(state[name]["count"]) for name in REPRESENTATIONS], dtype=np.int64),
        "point_action_norm_sq": point_action_norm_sq,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GEOMETRY_CACHE, **payload)
    return payload


def orient_eigenvectors(vectors: np.ndarray) -> np.ndarray:
    output = np.array(vectors, dtype=np.float64, copy=True)
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0.0:
            output[:, column] *= -1.0
    return output


def geometry_diagnostics(
    geometry: dict[str, np.ndarray],
    action_ids: list[str],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summaries: dict[str, dict[str, object]] = {}
    dimension_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    names = geometry["representation_names"].astype(str).tolist()
    for rep_index, name in enumerate(names):
        gram = np.asarray(geometry["grams"][rep_index], dtype=np.float64)
        sums = np.asarray(geometry["sums"][rep_index], dtype=np.float64)
        count = int(geometry["counts"][rep_index])
        diagonal = np.maximum(np.diag(gram), 0.0)
        denominator = np.sqrt(np.outer(diagonal, diagonal))
        cosine = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > TINY)
        covariance = gram - np.outer(sums, sums) / max(count, 1)
        covariance_diag = np.maximum(np.diag(covariance), 0.0)
        covariance_den = np.sqrt(np.outer(covariance_diag, covariance_diag))
        pearson = np.divide(covariance, covariance_den, out=np.zeros_like(covariance), where=covariance_den > TINY)

        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) * 0.5)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = orient_eigenvectors(eigenvectors[:, order])
        action_center = np.eye(N_DELTA) - np.ones((N_DELTA, N_DELTA)) / N_DELTA
        centered_gram = action_center @ gram @ action_center
        centered_values = np.maximum(np.linalg.eigvalsh((centered_gram + centered_gram.T) * 0.5)[::-1], 0.0)
        total = float(eigenvalues.sum())
        centered_total = float(centered_values.sum())
        uncentered_ratio = eigenvalues / max(total, TINY)
        centered_ratio = centered_values / max(centered_total, TINY)
        cumulative = np.cumsum(uncentered_ratio)
        centered_cumulative = np.cumsum(centered_ratio)
        effective_rank = total * total / max(float(np.square(eigenvalues).sum()), TINY)
        centered_effective_rank = centered_total * centered_total / max(float(np.square(centered_values).sum()), TINY)
        offdiag = ~np.eye(N_DELTA, dtype=bool)
        summaries[name] = {
            "gram": gram,
            "cosine": cosine,
            "pearson": pearson,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "uncentered_ratio": uncentered_ratio,
            "uncentered_cumulative": cumulative,
            "centered_eigenvalues": centered_values,
            "centered_ratio": centered_ratio,
            "centered_cumulative": centered_cumulative,
            "effective_rank": effective_rank,
            "centered_effective_rank": centered_effective_rank,
            "dimensions_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
            "dimensions_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
            "pair_cosine_mean": float(cosine[offdiag].mean()),
            "pair_abs_cosine_mean": float(np.abs(cosine[offdiag]).mean()),
            "pair_pearson_mean": float(pearson[offdiag].mean()),
            "pair_abs_pearson_mean": float(np.abs(pearson[offdiag]).mean()),
        }
        for dimension in range(1, N_DELTA + 1):
            dimension_rows.append(
                {
                    "representation": name,
                    "dimension": dimension,
                    "uncentered_explained": float(cumulative[dimension - 1]),
                    "centered_explained": float(centered_cumulative[dimension - 1]),
                    "uncentered_eigenvalue": float(eigenvalues[dimension - 1]),
                    "centered_eigenvalue": float(centered_values[dimension - 1]),
                    "effective_rank": effective_rank,
                    "centered_effective_rank": centered_effective_rank,
                }
            )
        for left in range(N_DELTA):
            for right in range(left + 1, N_DELTA):
                pair_rows.append(
                    {
                        "representation": name,
                        "left_action": action_ids[left],
                        "right_action": action_ids[right],
                        "cosine": float(cosine[left, right]),
                        "pearson": float(pearson[left, right]),
                    }
                )
    return summaries, dimension_rows, pair_rows


def build_scaled_basis(
    actions: list[e180.ActionSpec],
    provider: e180.ActionProvider,
    h_summary: dict[str, object],
    device: torch.device,
    batch_size: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues = np.asarray(h_summary["eigenvalues"], dtype=np.float64)
    eigenvectors = np.asarray(h_summary["eigenvectors"], dtype=np.float64)
    if np.any(eigenvalues[:N_DIRECTION] <= TINY):
        raise RuntimeError(f"degenerate top H eigenvalues: {eigenvalues[:N_DIRECTION]}")
    singular_values = np.sqrt(eigenvalues[:N_DIRECTION])
    raw_coordinates = eigenvectors[:, :N_DIRECTION] * singular_values[None, :]
    coordinate_scales = np.max(np.abs(raw_coordinates), axis=0)
    if np.any(coordinate_scales <= TINY):
        raise RuntimeError(f"degenerate coordinate scales: {coordinate_scales}")
    normalized_coordinates = raw_coordinates / coordinate_scales[None, :]

    if BASIS_H.is_file() and not force:
        basis = np.load(BASIS_H, mmap_mode="r")
        if basis.shape == (N_DIRECTION, *EXPECTED_SHAPE) and basis.dtype == np.complex64:
            return normalized_coordinates, coordinate_scales, eigenvectors[:, :N_DIRECTION]

    weights = eigenvectors[:, :N_DIRECTION] * (
        coordinate_scales / singular_values
    )[None, :]
    output = np.lib.format.open_memmap(
        BASIS_H,
        mode="w+",
        dtype=np.complex64,
        shape=(N_DIRECTION, *EXPECTED_SHAPE),
    )
    with torch.inference_mode():
        weight_tensor = torch.as_tensor(weights.T, dtype=torch.complex64, device=device)
        for start in range(0, N_POINT, batch_size):
            end = min(start + batch_size, N_POINT)
            baseline = torch.as_tensor(provider.block("V7", start, end), device=device)
            deltas = torch.stack(
                [
                    torch.as_tensor(provider.block(action.action_id, start, end), device=device) - baseline
                    for action in actions[1:]
                ],
                dim=0,
            )
            directions = torch.einsum("ka,abmns->kbmns", weight_tensor, deltas)
            output[:, start:end] = directions.cpu().numpy().astype(np.complex64)
            print(f"E182 basis points {end}/{N_POINT}", flush=True)
    output.flush()
    return normalized_coordinates, coordinate_scales, eigenvectors[:, :N_DIRECTION]


def utility_torch(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    truth_energy: torch.Tensor,
) -> torch.Tensor:
    batch = prediction.shape[0]
    shaped = prediction.reshape(batch, 2, 16, 8, 4, 192)
    pred_pas = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float().reshape(batch, 256, 4, 192)
    pas_num = (pred_pas * truth_pas).sum(dim=1, dtype=torch.float64)
    pas_den = (
        pred_pas.square().sum(dim=1, dtype=torch.float64)
        * truth_pas.square().sum(dim=1, dtype=torch.float64)
    ).clamp_min(0.0).sqrt()
    pas_den = torch.where(pas_den == 0.0, torch.full_like(pas_den, 1e-12), pas_den)
    pas = (pas_num / pas_den).mean(dim=(1, 2), dtype=torch.float64)

    pred_pdp = torch.fft.ifft(prediction, dim=-1).abs().square().float()
    pdp_num = (pred_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_den = (
        pred_pdp.square().sum(dim=3, dtype=torch.float64)
        * truth_pdp.square().sum(dim=3, dtype=torch.float64)
    ).clamp_min(0.0).sqrt()
    pdp_den = torch.where(pdp_den == 0.0, torch.full_like(pdp_den, 1e-12), pdp_den)
    pdp = (pdp_num / pdp_den).mean(dim=(1, 2), dtype=torch.float64)
    error = (prediction - truth).abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    point_nmse = error / truth_energy.clamp_min(1e-30)
    return 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + point_nmse)


def pas_pdp_torch(
    prediction: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = prediction.shape[0]
    shaped = prediction.reshape(batch, 2, 16, 8, 4, 192)
    pred_pas = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float().reshape(batch, 256, 4, 192)
    pas_num = (pred_pas * truth_pas).sum(dim=1, dtype=torch.float64)
    pas_den = (
        pred_pas.square().sum(dim=1, dtype=torch.float64)
        * truth_pas.square().sum(dim=1, dtype=torch.float64)
    ).clamp_min(0.0).sqrt()
    pas_den = torch.where(pas_den == 0.0, torch.full_like(pas_den, 1e-12), pas_den)
    pas = (pas_num / pas_den).mean(dim=(1, 2), dtype=torch.float64)

    pred_pdp = torch.fft.ifft(prediction, dim=-1).abs().square().float()
    pdp_num = (pred_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_den = (
        pred_pdp.square().sum(dim=3, dtype=torch.float64)
        * truth_pdp.square().sum(dim=3, dtype=torch.float64)
    ).clamp_min(0.0).sqrt()
    pdp_den = torch.where(pdp_den == 0.0, torch.full_like(pdp_den, 1e-12), pdp_den)
    pdp = (pdp_num / pdp_den).mean(dim=(1, 2), dtype=torch.float64)
    return pas, pdp


def project_coefficients(
    coefficients: torch.Tensor,
    point_gram: torch.Tensor,
    point_cap_sq: torch.Tensor,
    radius: float,
    nonzero_truth: torch.Tensor,
) -> torch.Tensor:
    value = coefficients.clamp(-1.0, 1.0)
    vector_norm = value.square().sum(dim=1).sqrt()
    radius_scale = torch.clamp(radius / vector_norm.clamp_min(1e-20), max=1.0)
    value = value * radius_scale[:, None]
    correction_sq = torch.einsum("bi,bij,bj->b", value, point_gram, value).clamp_min(0.0)
    cap_scale = torch.sqrt(point_cap_sq / correction_sq.clamp_min(1e-30)).clamp(max=1.0)
    value = value * cap_scale[:, None]
    return value * nonzero_truth[:, None]


def synthesize(
    baseline: torch.Tensor,
    directions: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    return baseline + torch.einsum(
        "bk,kbmns->bmns", coefficients.to(directions.dtype), directions
    )


def score_channel(path: Path, truth_all: np.ndarray, query_ids: np.ndarray, batch_size: int = 8) -> tuple[object, dict[str, float | int]]:
    prediction = np.load(path, mmap_mode="r")
    if prediction.shape != EXPECTED_SHAPE or prediction.dtype != np.complex64:
        raise ValueError(f"invalid oracle channel {path}: {prediction.shape}/{prediction.dtype}")
    accumulator = OfficialMetricAccumulator()
    for start in range(0, N_POINT, batch_size):
        end = min(start + batch_size, N_POINT)
        block = np.asarray(prediction[start:end], dtype=np.complex64)
        if not np.isfinite(block).all():
            raise ValueError(f"non-finite oracle rows {start}:{end}")
        accumulator.add(block, np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64))
    statistics = accumulator.statistics()
    return statistics, aggregate_statistics(statistics)


def optimize_continuous_oracles(
    query_ids: np.ndarray,
    actions: list[e180.ActionSpec],
    provider: e180.ActionProvider,
    normalized_action_coordinates: np.ndarray,
    point_action_norm_sq: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps: int,
    learning_rate: float,
    force: bool,
    e180_selection: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    truth_all = np.load(e180.TRAIN_CHANNEL, mmap_mode="r")
    basis = np.load(BASIS_H, mmap_mode="r")
    if basis.shape != (N_DIRECTION, *EXPECTED_SHAPE) or basis.dtype != np.complex64:
        raise ValueError(f"invalid basis: {basis.shape}/{basis.dtype}")

    baseline = torch.as_tensor(provider.block("V7", 0, N_POINT), device=device)
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    all_directions = torch.as_tensor(
        np.array(basis, dtype=np.complex64, copy=True), device=device
    )
    truth_pas, truth_pdp = action_spectra(truth)
    truth_energy_point = truth.abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    truth_energy_total = float(truth_energy_point.sum().item())
    nonzero_truth = (truth_energy_point > 1e-30).float()
    baseline_residual = baseline - truth
    baseline_error = baseline_residual.abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    with torch.no_grad():
        identity_pas, identity_pdp = pas_pdp_torch(baseline, truth_pas, truth_pdp)
        identity_nmse = float(baseline_error.sum().item()) / max(truth_energy_total, TINY)
        identity_score = (
            0.4 * float(identity_pas.mean().item())
            + 0.4 * float(identity_pdp.mean().item())
            + 0.2 / (1.0 + identity_nmse)
        )
    if abs(identity_score - EXPECTED_V7_SCORE) > 2e-5:
        raise RuntimeError(
            f"GPU official-objective V7 identity failed: {identity_score} != {EXPECTED_V7_SCORE}"
        )
    print(f"E182 full500 GPU V7 identity score={identity_score:.9f}", flush=True)

    previous_coefficients: np.ndarray | None = None
    all_coefficients: dict[str, np.ndarray] = {}
    oracle_rows: list[dict[str, object]] = []
    for dimension in range(1, N_DIRECTION + 1):
        path = ARTIFACT_DIR / ORACLE_H_PATTERN.format(dimension=dimension)
        coefficient_key = f"dim{dimension}"
        cached_coefficients: np.ndarray | None = None
        if COEFFICIENTS.is_file() and not force:
            cache = np.load(COEFFICIENTS, allow_pickle=False)
            if coefficient_key in cache and path.is_file():
                cached_coefficients = np.asarray(cache[coefficient_key], dtype=np.float32)
        if cached_coefficients is not None and cached_coefficients.shape == (N_POINT, dimension):
            coefficients = cached_coefficients
            print(f"E182 reuse dimension {dimension} oracle", flush=True)
        else:
            historical = np.asarray(normalized_action_coordinates[:, :dimension], dtype=np.float32)
            radius = float(np.max(np.linalg.norm(historical, axis=1)))
            directions = all_directions[:dimension]
            point_cap_sq = torch.as_tensor(
                np.max(point_action_norm_sq, axis=0), dtype=torch.float32, device=device
            )
            point_gram = torch.real(
                torch.einsum("kbmns,lbmns->bkl", directions.conj(), directions)
            ).float()
            linear_error = 2.0 * torch.real(
                torch.einsum(
                    "kbmns,bmns->bk", directions.conj(), baseline_residual
                )
            ).float()

            def constrained_value(value: torch.Tensor) -> torch.Tensor:
                return project_coefficients(
                    value, point_gram, point_cap_sq, radius, nonzero_truth
                )

            def error_per_point(value: torch.Tensor, start: int = 0) -> torch.Tensor:
                end = start + len(value)
                return (
                    baseline_error[start:end]
                    + (value * linear_error[start:end]).sum(dim=1)
                    + torch.einsum(
                        "bi,bij,bj->b", value, point_gram[start:end], value
                    )
                ).clamp_min(0.0)

            def evaluate_official(value: torch.Tensor) -> tuple[float, torch.Tensor]:
                with torch.no_grad():
                    constrained = constrained_value(value)
                    pas_sum = 0.0
                    pdp_sum = 0.0
                    for local_start in range(0, N_POINT, batch_size):
                        local_end = min(local_start + batch_size, N_POINT)
                        prediction = synthesize(
                            baseline[local_start:local_end],
                            directions[:, local_start:local_end],
                            constrained[local_start:local_end],
                        )
                        pas, pdp = pas_pdp_torch(
                            prediction,
                            truth_pas[local_start:local_end],
                            truth_pdp[local_start:local_end],
                        )
                        pas_sum += float(pas.sum().item())
                        pdp_sum += float(pdp.sum().item())
                    error_total = float(error_per_point(constrained).sum().item())
                    score = (
                        0.4 * pas_sum / N_POINT
                        + 0.4 * pdp_sum / N_POINT
                        + 0.2 / (1.0 + error_total / max(truth_energy_total, TINY))
                    )
                    return score, constrained

            candidates: list[torch.Tensor] = [
                torch.zeros((N_POINT, dimension), dtype=torch.float32, device=device)
            ]
            if previous_coefficients is not None:
                prior = np.zeros((N_POINT, dimension), dtype=np.float32)
                prior[:, : dimension - 1] = previous_coefficients
                candidates.append(torch.as_tensor(prior, device=device))
            selected_projection = np.zeros((N_POINT, dimension), dtype=np.float32)
            selected_mask = e180_selection > 0
            selected_projection[selected_mask] = historical[e180_selection[selected_mask] - 1]
            candidates.append(torch.as_tensor(selected_projection, device=device))
            for action_coordinate in historical:
                candidates.append(
                    torch.as_tensor(action_coordinate, device=device).repeat(N_POINT, 1)
                )

            best_score = -float("inf")
            best_x = candidates[0]
            for candidate in candidates:
                candidate_score, constrained = evaluate_official(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_x = constrained.clone()

            parameter = torch.nn.Parameter(best_x.clone())
            optimizer = torch.optim.Adam([parameter], lr=learning_rate)
            for step in range(steps):
                with torch.no_grad():
                    current = constrained_value(parameter)
                    error_total = error_per_point(current).sum()
                    nmse = error_total / max(truth_energy_total, TINY)
                    error_weight = float(
                        0.2 / max(truth_energy_total, TINY) / (1.0 + nmse.item()) ** 2
                    )

                optimizer.zero_grad(set_to_none=True)
                pas_sum = 0.0
                pdp_sum = 0.0
                for local_start in range(0, N_POINT, batch_size):
                    local_end = min(local_start + batch_size, N_POINT)
                    constrained = project_coefficients(
                        parameter[local_start:local_end],
                        point_gram[local_start:local_end],
                        point_cap_sq[local_start:local_end],
                        radius,
                        nonzero_truth[local_start:local_end],
                    )
                    prediction = synthesize(
                        baseline[local_start:local_end],
                        directions[:, local_start:local_end],
                        constrained,
                    )
                    pas, pdp = pas_pdp_torch(
                        prediction,
                        truth_pas[local_start:local_end],
                        truth_pdp[local_start:local_end],
                    )
                    batch_error = error_per_point(constrained, local_start)
                    loss = (
                        -0.4 * pas.sum() / N_POINT
                        -0.4 * pdp.sum() / N_POINT
                        + error_weight * batch_error.sum()
                    )
                    loss.backward()
                    pas_sum += float(pas.detach().sum().item())
                    pdp_sum += float(pdp.detach().sum().item())
                optimizer.step()
                with torch.no_grad():
                    parameter.copy_(constrained_value(parameter))
                    current_error = float(error_per_point(current).sum().item())
                    current_score = (
                        0.4 * pas_sum / N_POINT
                        + 0.4 * pdp_sum / N_POINT
                        + 0.2 / (1.0 + current_error / max(truth_energy_total, TINY))
                    )
                    if current_score > best_score:
                        best_score = current_score
                        best_x = current.clone()
                if (step + 1) % 20 == 0:
                    print(
                        f"E182 official oracle dim={dimension} step={step + 1}/{steps} score={best_score:.6f}",
                        flush=True,
                    )

            final_score, final_x = evaluate_official(parameter)
            if final_score > best_score:
                best_score = final_score
                best_x = final_x.clone()
            coefficients = best_x.cpu().numpy().astype(np.float32)
            output = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.complex64, shape=EXPECTED_SHAPE
            )
            with torch.no_grad():
                for local_start in range(0, N_POINT, batch_size):
                    local_end = min(local_start + batch_size, N_POINT)
                    prediction = synthesize(
                        baseline[local_start:local_end],
                        directions[:, local_start:local_end],
                        best_x[local_start:local_end],
                    )
                    output[local_start:local_end] = prediction.cpu().numpy().astype(np.complex64)
            output.flush()
            print(
                f"E182 official oracle dim={dimension} optimized score={best_score:.6f}",
                flush=True,
            )

        all_coefficients[coefficient_key] = coefficients
        previous_coefficients = coefficients
        _, official = score_channel(path, truth_all, query_ids)
        oracle_rows.append(
            {
                "method": f"bounded_continuous_{dimension}d",
                "dimension": dimension,
                **official,
                "delta_v7": float(official["score"]) - EXPECTED_V7_SCORE,
                "delta_e180": float(official["score"]) - EXPECTED_E180_SCORE,
                "delta_expanded46": float(official["score"]) - EXPECTED_EXPANDED_SCORE,
                "coefficient_radius_max": float(
                    np.max(np.linalg.norm(normalized_action_coordinates[:, :dimension], axis=1))
                ),
                "coefficient_abs_max": float(np.max(np.abs(coefficients))),
                "active_fraction": float(np.mean(np.linalg.norm(coefficients, axis=1) > 1e-6)),
            }
        )
        np.savez_compressed(COEFFICIENTS, **all_coefficients)
    return oracle_rows, all_coefficients


def selected_action_projection(
    actions: list[e180.ActionSpec],
    provider: e180.ActionProvider,
    normalized_coordinates: np.ndarray,
    point_action_norm_sq: np.ndarray,
) -> dict[str, object]:
    metrics = e180.load_cached_metrics(actions)
    action_ids = [action.action_id for action in actions]
    if set(metrics) != set(action_ids):
        raise RuntimeError("primary E180 metrics are incomplete")
    selection, _, aggregate = e180.oracle_selection(action_ids, metrics)
    if abs(float(aggregate["score"]) - EXPECTED_E180_SCORE) > 2e-5:
        raise RuntimeError(f"E180 oracle identity failed: {aggregate}")

    basis = np.load(BASIS_H, mmap_mode="r")
    selected_total = 0.0
    selected_residual_energy = np.zeros(N_DIRECTION, dtype=np.float64)
    for point in range(N_POINT):
        selected = int(selection[point])
        if selected == 0:
            continue
        baseline = provider.block("V7", point, point + 1)[0]
        selected_delta = (
            provider.block(actions[selected].action_id, point, point + 1)[0] - baseline
        )
        total_energy = float(point_action_norm_sq[selected - 1, point])
        selected_total += total_energy
        for dimension in range(1, N_DIRECTION + 1):
            projected = np.einsum(
                "k,k...->...",
                normalized_coordinates[selected - 1, :dimension],
                np.asarray(basis[:dimension, point], dtype=np.complex64),
                optimize=True,
            )
            selected_residual_energy[dimension - 1] += float(
                np.sum(np.abs(selected_delta - projected) ** 2, dtype=np.float64)
            )
    return {
        "e180_oracle_score_identity": aggregate,
        "selection": selection,
        "selected_total_correction_energy": selected_total,
        "selected_pointwise_correction_explained": 1.0
        - selected_residual_energy / max(selected_total, TINY),
    }


def make_plots(
    summaries: dict[str, dict[str, object]],
    action_ids: list[str],
    oracle_rows: list[dict[str, object]],
) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [value.replace("carrier_", "c") for value in action_ids]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    for axis, name, title in zip(
        axes,
        ("delta_h", "delta_pas_l2_shape", "delta_pdp_l2_shape"),
        ("Delta H", "Delta PAS shape", "Delta PDP shape"),
    ):
        matrix = np.asarray(summaries[name]["cosine"])
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_title(title)
        axis.set_xticks(range(N_DELTA), labels=labels, rotation=90, fontsize=7)
        axis.set_yticks(range(N_DELTA), labels=labels, fontsize=7)
    fig.colorbar(image, ax=axes, shrink=0.8, label="cosine")
    fig.savefig(ANALYSIS_DIR / "动作差分相似度.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for name, label in (
        ("delta_h", "Delta H"),
        ("delta_pas_l2_shape", "Delta PAS shape"),
        ("delta_pdp_l2_shape", "Delta PDP shape"),
    ):
        values = np.asarray(summaries[name]["uncentered_cumulative"])
        axis.plot(np.arange(1, N_DELTA + 1), values, marker="o", label=label)
    axis.axhline(0.90, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Number of directions", ylabel="Cumulative correction energy", ylim=(0.0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(ANALYSIS_DIR / "低维方向累计解释率.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    dimensions = [int(row["dimension"]) for row in oracle_rows]
    scores = [float(row["score"]) for row in oracle_rows]
    axis.plot(dimensions, scores, marker="o", linewidth=2, label="bounded continuous oracle")
    axis.axhline(EXPECTED_V7_SCORE, color="gray", linestyle=":", label="V7")
    axis.axhline(EXPECTED_E180_SCORE, color="tab:orange", linestyle="--", label="13-action select-one")
    axis.axhline(EXPECTED_EXPANDED_SCORE, color="tab:red", linestyle="--", label="46-action select-one")
    axis.set(xlabel="Continuous dimensions", ylabel="Official fixed500 score", xticks=dimensions)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(ANALYSIS_DIR / "连续动作oracle官方分.png", dpi=180)
    plt.close(fig)


def write_report(
    actions: list[e180.ActionSpec],
    summaries: dict[str, dict[str, object]],
    oracle_rows: list[dict[str, object]],
    projection: dict[str, object],
    timing: dict[str, float],
    peak_memory_gib: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    h_explained = np.asarray(summaries["delta_h"]["uncentered_cumulative"])
    success = bool(
        h_explained[3] >= SUCCESS_EXPLAINED_4D
        and float(oracle_rows[3]["score"]) >= SUCCESS_SCORE_4D
    )
    payload = {
        "version": VERSION,
        "status": "diagnostic_only_no_candidate",
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "action_ids": [action.action_id for action in actions],
        "difference_action_ids": [action.action_id for action in actions[1:]],
        "representations": summaries,
        "continuous_oracles": oracle_rows,
        "selected_action_projection": projection,
        "baselines": {
            "V7": EXPECTED_V7_SCORE,
            "E180_13_action_select_one": EXPECTED_E180_SCORE,
            "E180_46_action_select_one": EXPECTED_EXPANDED_SCORE,
        },
        "decision": {
            "four_dimensional_h_explained_threshold": SUCCESS_EXPLAINED_4D,
            "four_dimensional_score_threshold": SUCCESS_SCORE_4D,
            "two_dimensional_score_threshold": SUCCESS_SCORE_2D,
            "four_dimensional_h_explained": float(h_explained[3]),
            "four_dimensional_score": float(oracle_rows[3]["score"]),
            "two_dimensional_score": float(oracle_rows[1]["score"]),
            "few_continuous_directions_supported": success,
        },
        "constraints": {
            "real_coefficients_only": True,
            "zero_action_available": True,
            "per_component_historical_bound": True,
            "historical_normalized_radius_bound": True,
            "per_point_historical_delta_h_norm_cap": True,
            "zero_truth_forced_to_v7": True,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
        },
        "timing_seconds": timing,
        "device": str(args.device),
        "peak_gpu_memory_gib": peak_memory_gib,
    }
    REPORT_JSON.write_text(
        json.dumps(json_compatible(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    representation_labels = {
        "delta_h": "完整复H",
        "delta_pas_power": "原始PAS功率",
        "delta_pdp_power": "原始PDP功率",
        "delta_pas_l2_shape": "PAS官方形状",
        "delta_pdp_l2_shape": "PDP官方形状",
        "delta_pas_sqrt_shape": "PAS开方形状",
        "delta_pdp_sqrt_shape": "PDP开方形状",
    }
    lines = [
        "# E182：Clean动作是离散选择还是少数连续方向",
        "",
        "本实验只使用E180冻结的13个Clean完整H动作。V7为零修正，以下维数分析针对其余12个动作相对V7的实际变化。",
        "",
        "## 低维解释率",
        "",
        "| 差分表示 | 1维 | 2维 | 3维 | 4维 | 90%所需维数 | 有效秩 | 平均绝对cosine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in REPRESENTATIONS:
        summary = summaries[name]
        cumulative = np.asarray(summary["uncentered_cumulative"])
        lines.append(
            f"| {representation_labels[name]} | {cumulative[0]:.3%} | {cumulative[1]:.3%} | "
            f"{cumulative[2]:.3%} | {cumulative[3]:.3%} | {int(summary['dimensions_90pct'])} | "
            f"{float(summary['effective_rank']):.3f} | {float(summary['pair_abs_cosine_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "这里的1/2/3/4维解释率表示：只保留这么多个共同方向，可以重建多少历史动作修正能量。完整H使用复数张量的实内积，因此连续系数只能调动作方向的真实强弱，不能任意旋转相位。",
            "",
            "## 受约束连续oracle",
            "",
            "| 方法 | PAS | PDP | NMSE | 总分 | 相对V7 | 相对13动作选1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| V7 | 0.597983 | 0.756446 | 0.947710 | {EXPECTED_V7_SCORE:.6f} | +0.000000 | {EXPECTED_V7_SCORE - EXPECTED_E180_SCORE:+.6f} |",
        ]
    )
    for row in oracle_rows:
        lines.append(
            f"| {int(row['dimension'])}维连续 | {float(row['pas']):.6f} | {float(row['pdp']):.6f} | "
            f"{float(row['nmse']):.6f} | {float(row['score']):.6f} | {float(row['delta_v7']):+.6f} | "
            f"{float(row['delta_e180']):+.6f} |"
        )
    lines.extend(
        [
            f"| 13动作选1 | 0.657713 | 0.772667 | 0.903876 | {EXPECTED_E180_SCORE:.6f} | {EXPECTED_E180_SCORE - EXPECTED_V7_SCORE:+.6f} | +0.000000 |",
            f"| 46动作选1 | - | - | - | {EXPECTED_EXPANDED_SCORE:.6f} | {EXPECTED_EXPANDED_SCORE - EXPECTED_V7_SCORE:+.6f} | {EXPECTED_EXPANDED_SCORE - EXPECTED_E180_SCORE:+.6f} |",
            "",
            "连续oracle每点可看GT选择系数，但系数允许为0、每维不超过历史投影范围、总系数半径不超过历史动作、每点Delta H范数不超过该点12个历史动作的最大值。它是动作空间上限，不是可提交算法。",
            "",
            "## 裁决",
            "",
            (
                "预注册的少数连续方向假设通过：前4个完整H方向达到解释率和完整分双门。"
                if success
                else "预注册的少数连续方向假设未通过：前4个完整H方向没有同时达到90%解释率和13动作选1完整分。"
            ),
            "",
            f"2维连续oracle是否达到0.660000：{'是' if float(oracle_rows[1]['score']) >= SUCCESS_SCORE_2D else '否'}。",
            "",
            f"实测总耗时{timing['total']:.1f}s，设备{args.device}，PyTorch峰值GPU显存约{peak_memory_gib:.2f}GiB。",
            "",
            "本轮到此停止，不训练系数预测器，不生成提交版本。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E182 requires one CUDA GPU")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    query_ids, fold_id, site, actions, provider = load_clean_actions()
    print("E182 Clean lineage and 13-action identity gate passed", flush=True)
    if args.dry_run:
        print({"query_count": len(query_ids), "folds": np.unique(fold_id).tolist(), "sites": np.unique(site).tolist(), "actions": [a.action_id for a in actions]})
        return

    geometry_started = time.perf_counter()
    geometry = accumulate_geometry(
        actions, provider, device, args.geometry_batch_size, args.force_geometry
    )
    geometry_seconds = time.perf_counter() - geometry_started
    action_ids = [action.action_id for action in actions[1:]]
    summaries, dimension_rows, pair_rows = geometry_diagnostics(geometry, action_ids)
    write_csv(DIMENSION_CSV, dimension_rows)
    write_csv(PAIR_CSV, pair_rows)

    basis_started = time.perf_counter()
    normalized_coordinates, coordinate_scales, _ = build_scaled_basis(
        actions,
        provider,
        summaries["delta_h"],
        device,
        args.geometry_batch_size,
        args.force_basis,
    )
    basis_seconds = time.perf_counter() - basis_started
    coordinate_rows = []
    for action_index, action_id in enumerate(action_ids):
        row: dict[str, object] = {"action_id": action_id}
        for dimension in range(N_DIRECTION):
            row[f"normalized_coordinate_{dimension + 1}"] = float(normalized_coordinates[action_index, dimension])
            row[f"raw_coordinate_{dimension + 1}"] = float(
                normalized_coordinates[action_index, dimension] * coordinate_scales[dimension]
            )
        coordinate_rows.append(row)
    write_csv(COORDINATE_CSV, coordinate_rows)

    primary_metrics = e180.load_cached_metrics(actions)
    e180_selection, _, e180_identity = e180.oracle_selection(
        [action.action_id for action in actions], primary_metrics
    )
    if abs(float(e180_identity["score"]) - EXPECTED_E180_SCORE) > 2e-5:
        raise RuntimeError(f"E180 oracle identity failed before continuous optimization: {e180_identity}")

    oracle_started = time.perf_counter()
    oracle_rows, _ = optimize_continuous_oracles(
        query_ids,
        actions,
        provider,
        normalized_coordinates,
        np.asarray(geometry["point_action_norm_sq"], dtype=np.float64),
        device,
        args.opt_batch_size,
        args.steps,
        args.learning_rate,
        args.force_oracle,
        e180_selection,
    )
    oracle_seconds = time.perf_counter() - oracle_started
    write_csv(ORACLE_CSV, oracle_rows)

    projection = selected_action_projection(
        actions,
        provider,
        normalized_coordinates,
        np.asarray(geometry["point_action_norm_sq"], dtype=np.float64),
    )
    make_plots(summaries, action_ids, oracle_rows)
    total_seconds = time.perf_counter() - started
    timing = {
        "geometry": geometry_seconds,
        "basis": basis_seconds,
        "continuous_oracles_and_scoring": oracle_seconds,
        "total": total_seconds,
    }
    peak_gpu_memory_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    payload = write_report(
        actions,
        summaries,
        oracle_rows,
        projection,
        timing,
        peak_gpu_memory_gib,
        args,
    )
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    payload["peak_rss_gib"] = peak_rss_gib
    REPORT_JSON.write_text(
        json.dumps(json_compatible(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_compatible(payload["decision"]), ensure_ascii=False, indent=2), flush=True)
    print(f"E182 complete in {total_seconds:.1f}s; peak GPU {peak_gpu_memory_gib:.2f} GiB; RSS {peak_rss_gib:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
