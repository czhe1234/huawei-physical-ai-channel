#!/usr/bin/env python3
"""Replay and summarize Round 2 versions on one canonical 500-point OOF set.

V1-V8 already have complete 500-point evidence in their version reports.  V9
is reconstructed as a control, then V10 and V11 are replayed for all 500 OOF
queries, including the zero-channel points omitted by their original module
diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

from analyze_joint_path_token_oracle import OOF_META, v9_channel
from analyze_v11_low_rank_path_pas_residual import (
    normalize_pas,
    pas_from_slices,
    pas_slices,
)
from analyze_v11_reliable_sqrt_pas_gate import percentile_rank
from analyze_v11_sqrt_pas_source_consensus import build_consensus
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from common.official_metrics import (
    OfficialMetricAccumulator,
    aggregate_statistics,
)
from project_paths import PATHS
from train_full_path_pas_slice_gate import project_joint
from train_full_path_pdp_link_gate import FOLD_IDS, load_model, predict_alpha
from train_path_pdp_link_gate import link_coordinates, project


VERSION = "Round2-E77-canonical-500-version-replay"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "canonical_500_version_replay"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_canonical_500_version_replay"
)
JSON_REPORT = ANALYSIS_DIR / "canonical_500_version_scores.json"
CSV_REPORT = ANALYSIS_DIR / "canonical_500_version_scores.csv"
POINT_METRICS = ARTIFACT_DIR / "canonical_v9_v11_point_metrics.npz"

V9_CHANNEL = ARTIFACT_DIR / "V9_Canonical_500_OOF_Channel.npy"
V10_CHANNEL = ARTIFACT_DIR / "V10_Canonical_500_OOF_Channel.npy"
V11_CHANNEL = ARTIFACT_DIR / "V11_Canonical_500_OOF_Channel.npy"
V9_PDP = ARTIFACT_DIR / "v9_base_pdp_f16.npy"
V10_ALPHA = ARTIFACT_DIR / "v10_cross_fitted_alpha_f16.npy"
V11_GATE_FEATURES = ARTIFACT_DIR / "v11_pair_rnorm_raw_features.npy"
V11_GATE = ARTIFACT_DIR / "v11_pair_rnorm_gate_f16.npy"

FULL_PATH_DIR = PATHS.artifacts / "round2" / "full_path_pdp_link_gate"
RAW_PATH_PDP = FULL_PATH_DIR / "train_raw_path_pdp_f16.npy"
POINT_CONTEXT = FULL_PATH_DIR / "train_point_context.npy"

E40_PAS = (
    PATHS.artifacts
    / "round2"
    / "sionna_official_calibration"
    / "oof_pas_mean_f16.npy"
)
V9_ENDPOINT_PAS = (
    PATHS.artifacts
    / "round2"
    / "v9_joint_sionna_endpoints"
    / "oof_pas_joint_f16.npy"
)
E39_BASE_PAS = (
    PATHS.artifacts
    / "round2"
    / "sionna_multiseed"
    / "primary_d16_base_v4lambda_1m"
    / "full"
    / "checkpoints"
    / "n24_train_pas_mean.npy"
)
E39_SCATTER_PAS = (
    PATHS.artifacts
    / "round2"
    / "sionna_multiseed"
    / "primary_d10_scatter015_v4lambda_1m"
    / "full"
    / "checkpoints"
    / "n24_train_pas_mean.npy"
)

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
TINY = 1e-30
EXPECTED_SHAPE = (500, 256, 4, 192)
V10_POLICY = {"alpha_scale": 0.50, "order": "pas_pdp", "cycles": 2}
V11_POLICY = {
    "gate": "pair_rnorm",
    "quantile": 0.70,
    "alpha": 0.20,
    "order": "pdp_pas",
    "cycles": 4,
}

PARENTS = {
    "V1": "Foundation",
    "V2": "V1",
    "V3": "V2",
    "V4": "V3",
    "V5": "V4",
    "V6": "V4",
    "V7": "V6",
    "V8": "V7",
    "V9": "V8",
    "V10": "V9",
    "V11": "V10",
}

ONLINE_SCORES = {
    "V1": 0.594822,
    "V2": 0.628977,
    "V3": 0.629603,
    "V4": 0.636211,
    "V5": None,
    "V6": 0.636556,
    "V7": 0.636929,
    "V8": 0.662169,
    "V9": 0.675445,
    "V10": 0.679708,
    "V11": 0.682145,
}

# These are strict end-to-end 500-point OOF estimates.  For V5-V7 the nested
# result is used so action selection for a held hole never sees that hole.
HISTORICAL_METRICS = {
    "V1": (
        PATHS.analysis
        / "results"
        / "version_reports"
        / "V1"
        / "V1_complete_pre_sionna_pipeline.json",
        ("local_validation",),
        "500-point OOF final prediction",
    ),
    "V2": (
        PATHS.analysis
        / "results"
        / "version_reports"
        / "V2"
        / "V2_rbf_sqrt_spectral_field.json",
        ("local_validation",),
        "500-point OOF final prediction",
    ),
    "V3": (
        PATHS.analysis
        / "results"
        / "version_reports"
        / "V3"
        / "V3_anchored_neural_rbf.json",
        ("local_validation",),
        "500-point OOF final prediction",
    ),
    "V4": (
        PATHS.analysis
        / "results"
        / "version_reports"
        / "V4"
        / "V4_predictability_calibrated_exact_projection.json",
        ("local_validation",),
        "500-point OOF nested final prediction",
    ),
    "V5": (
        PATHS.analysis
        / "results"
        / "version_reports"
        / "V5"
        / "V5_sionna_shared_spectral_calibration.json",
        ("local_validation", "nested_policy"),
        "500-point OOF nested policy",
    ),
    "V6": (
        PATHS.analysis
        / "results"
        / "version_reports"
        / "V6"
        / "V6_regime_component_lifecycle.json",
        ("local_validation", "nested_policy"),
        "500-point OOF nested policy",
    ),
    "V7": (
        PATHS.analysis
        / "results"
        / "version_reports"
        / "V7"
        / "V7_complex_carrier_lifecycle.json",
        ("local_validation", "nested_lifecycle"),
        "500-point OOF nested lifecycle",
    ),
    "V8": (
        PATHS.analysis
        / "results"
        / "version_reports"
        / "V8"
        / "V8_full_sionna_migration.json",
        ("local_validation", "fixed_policy"),
        "500-point OOF frozen policy on nested V7",
    ),
}

FOUNDATION_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_pre_sionna_pipeline"
    / "round2_pre_sionna_pipeline.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-historical-control-assertions",
        action="store_true",
        help=(
            "Record historical replay differences without requiring equality. "
            "Use only when evaluating an intentionally changed V9/V10 pipeline."
        ),
    )
    return parser.parse_args()


def nested_value(payload: dict[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"missing report key {'.'.join(keys)}")
        current = current[key]
    if not isinstance(current, dict):
        raise TypeError(f"metric at {'.'.join(keys)} is not an object")
    return current


def canonical_metric(metric: dict[str, object]) -> dict[str, float | int]:
    required = ("count", "pas", "pdp", "nmse", "nmse_score", "score")
    missing = [key for key in required if key not in metric]
    if missing:
        raise KeyError(f"metric is missing {missing}")
    result: dict[str, float | int] = {"count": int(metric["count"])}
    result.update({key: float(metric[key]) for key in required[1:]})
    if result["count"] != 500:
        raise ValueError(f"canonical metric must contain 500 points, got {result['count']}")
    return result


def validate_channel(path: Path) -> np.memmap:
    channel = np.load(path, mmap_mode="r")
    if channel.shape != EXPECTED_SHAPE or channel.dtype != np.complex64:
        raise ValueError(f"invalid channel {PATHS.relative(path)}: {channel.shape}/{channel.dtype}")
    for start in range(0, len(channel), 8):
        value = np.asarray(channel[start : start + 8])
        if not np.isfinite(value.real).all() or not np.isfinite(value.imag).all():
            raise ValueError(f"non-finite channel values in {PATHS.relative(path)}")
    return channel


def build_v9(device: torch.device, batch_size: int, force: bool) -> np.memmap:
    if V9_CHANNEL.is_file() and not force:
        return validate_channel(V9_CHANNEL)
    print("Reconstructing V9 for all 500 OOF points", flush=True)
    prediction = v9_channel(np.arange(500, dtype=np.int64), device, batch_size)
    if prediction.shape != EXPECTED_SHAPE:
        raise ValueError(f"unexpected V9 shape: {prediction.shape}")
    np.save(V9_CHANNEL, prediction.astype(np.complex64, copy=False))
    del prediction
    return validate_channel(V9_CHANNEL)


def build_v9_pdp(
    v9: np.ndarray, device: torch.device, batch_size: int, force: bool
) -> np.memmap:
    if V9_PDP.is_file() and not force:
        value = np.load(V9_PDP, mmap_mode="r")
        if value.shape == EXPECTED_SHAPE and value.dtype == np.float16:
            return value
    output = np.lib.format.open_memmap(
        V9_PDP, mode="w+", dtype=np.float16, shape=EXPECTED_SHAPE
    )
    for start in range(0, len(v9), batch_size):
        end = min(start + batch_size, len(v9))
        channel = torch.as_tensor(
            np.array(v9[start:end], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            output[start:end] = pdp_distribution(channel).cpu().numpy().astype(np.float16)
        if end % 100 == 0 or end == len(v9):
            output.flush()
            print(f"V9 PDP {end}/{len(v9)}", flush=True)
    output.flush()
    return np.load(V9_PDP, mmap_mode="r")


def predict_v10_alpha(
    query_ids: np.ndarray,
    base_pdp: np.ndarray,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if V10_ALPHA.is_file() and not force:
        cached = np.load(V10_ALPHA, mmap_mode="r")
        if cached.shape == (500, 1024):
            return cached

    raw_all = np.load(RAW_PATH_PDP, mmap_mode="r")
    context_all = np.load(POINT_CONTEXT, mmap_mode="r")
    broad_fold = np.asarray(np.load(FOLD_IDS, mmap_mode="r")[query_ids], dtype=np.int8)
    link_tensor = torch.as_tensor(link_coordinates(), device=device)
    alpha = np.empty((500, 1024), dtype=np.float32)
    for fold in range(5):
        rows = np.flatnonzero(broad_fold == fold)
        model, checkpoint = load_model(fold, device)
        local_base = np.asarray(base_pdp[rows])
        local_raw = np.asarray(raw_all[query_ids[rows]])
        local_context = np.asarray(context_all[query_ids[rows]], dtype=np.float32)
        alpha[rows] = predict_alpha(
            model,
            np.arange(len(rows), dtype=np.int64),
            local_base,
            local_raw,
            local_context,
            np.asarray(checkpoint["context_mean"], dtype=np.float32),
            np.asarray(checkpoint["context_std"], dtype=np.float32),
            link_tensor,
            device,
            batch_size,
        )
        print(f"V10 OOF alpha fold {fold + 1}/5 ({len(rows)} points)", flush=True)
        del model, local_base, local_raw, local_context
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.save(V10_ALPHA, alpha.astype(np.float16))
    return np.load(V10_ALPHA, mmap_mode="r")


def build_v10(
    v9: np.ndarray,
    query_ids: np.ndarray,
    base_pdp: np.ndarray,
    alpha: np.ndarray,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> np.memmap:
    if V10_CHANNEL.is_file() and not force:
        return validate_channel(V10_CHANNEL)
    raw_all = np.load(RAW_PATH_PDP, mmap_mode="r")
    output = np.lib.format.open_memmap(
        V10_CHANNEL, mode="w+", dtype=np.complex64, shape=EXPECTED_SHAPE
    )
    for start in range(0, len(v9), batch_size):
        end = min(start + batch_size, len(v9))
        reference = torch.as_tensor(
            np.array(v9[start:end], dtype=np.complex64, copy=True), device=device
        )
        local_base = torch.as_tensor(
            np.asarray(base_pdp[start:end], dtype=np.float32), device=device
        )
        local_raw = torch.as_tensor(
            np.asarray(raw_all[query_ids[start:end]], dtype=np.float32), device=device
        )
        local_alpha = torch.clamp(
            V10_POLICY["alpha_scale"]
            * torch.as_tensor(
                np.asarray(alpha[start:end], dtype=np.float32), device=device
            ),
            0.0,
            1.0,
        ).reshape(end - start, 256, 4)
        desired = (
            (1.0 - local_alpha[..., None]) * local_base
            + local_alpha[..., None] * local_raw
        )
        desired /= desired.sum(dim=3, keepdim=True).clamp_min(TINY)
        with torch.inference_mode():
            candidate = project(
                reference,
                desired,
                int(V10_POLICY["cycles"]),
                str(V10_POLICY["order"]),
            )
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        if end % 100 == 0 or end == len(v9):
            output.flush()
            print(f"V10 canonical projection {end}/{len(v9)}", flush=True)
    output.flush()
    return validate_channel(V10_CHANNEL)


def pas_sources() -> tuple[np.ndarray, ...]:
    return tuple(
        np.load(path, mmap_mode="r")
        for path in (E40_PAS, V9_ENDPOINT_PAS, E39_BASE_PAS, E39_SCATTER_PAS)
    )


def source_amplitudes(
    arrays: tuple[np.ndarray, ...],
    query_ids: np.ndarray,
    start: int,
    end: int,
    device: torch.device,
) -> list[torch.Tensor]:
    values = [
        np.asarray(arrays[0][start:end], dtype=np.float32),
        np.asarray(arrays[1][start:end], dtype=np.float32),
        np.asarray(arrays[2][query_ids[start:end]], dtype=np.float32),
        np.asarray(arrays[3][query_ids[start:end]], dtype=np.float32),
    ]
    values[2] = np.roll(values[2], 2, axis=3)
    values[3] = np.roll(values[3], 2, axis=3)
    output = []
    for value in values:
        tensor = normalize_pas(
            torch.as_tensor(np.array(value, copy=True), device=device)
        )
        output.append(torch.sqrt(pas_slices(tensor).clamp_min(0.0)))
    return output


def base_pas_amplitude(
    base: np.ndarray, start: int, end: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    channel = torch.as_tensor(
        np.array(base[start:end], dtype=np.complex64, copy=True), device=device
    )
    pas = normalize_pas(pas_distribution(channel))
    return channel, torch.sqrt(pas_slices(pas).clamp_min(0.0))


def build_v11_gate(
    v10: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if V11_GATE.is_file() and not force:
        cached = np.load(V11_GATE, mmap_mode="r")
        if cached.shape == (500, 4, 192):
            return cached
    raw = np.lib.format.open_memmap(
        V11_GATE_FEATURES,
        mode="w+",
        dtype=np.float32,
        shape=(500, 4, 192, 2),
    )
    for start in range(0, len(v10), batch_size):
        end = min(start + batch_size, len(v10))
        _, local_base = base_pas_amplitude(v10, start, end, device)
        sources = source_amplitudes(arrays, query_ids, start, end, device)
        residual = torch.stack([source - local_base for source in sources])
        residual_norm = residual.square().sum(dim=4).sqrt()
        pair_spread = torch.stack([source.square() for source in sources]).std(
            dim=0
        ).mean(dim=3)
        raw[start:end, :, :, 0] = pair_spread.cpu().numpy()
        raw[start:end, :, :, 1] = residual_norm.median(dim=0).values.cpu().numpy()
        if end % 100 == 0 or end == len(v10):
            raw.flush()
            print(f"V11 reliability features {end}/{len(v10)}", flush=True)
    raw.flush()
    raw_tensor = torch.as_tensor(np.asarray(raw), device=device)
    score = 0.5 * (
        percentile_rank(raw_tensor[..., 0]) + percentile_rank(raw_tensor[..., 1])
    )
    gate = (score <= float(V11_POLICY["quantile"])).float()
    np.save(V11_GATE, gate.cpu().numpy().astype(np.float16))
    return np.load(V11_GATE, mmap_mode="r")


def build_v11(
    v10: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    query_ids: np.ndarray,
    gate: np.ndarray,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> np.memmap:
    if V11_CHANNEL.is_file() and not force:
        return validate_channel(V11_CHANNEL)
    output = np.lib.format.open_memmap(
        V11_CHANNEL, mode="w+", dtype=np.complex64, shape=EXPECTED_SHAPE
    )
    for start in range(0, len(v10), batch_size):
        end = min(start + batch_size, len(v10))
        reference, local_base = base_pas_amplitude(v10, start, end, device)
        sources = source_amplitudes(arrays, query_ids, start, end, device)
        consensus = build_consensus(local_base, sources, tuple(range(4)))
        local_gate = torch.as_tensor(
            np.asarray(gate[start:end], dtype=np.float32), device=device
        )
        amplitude = (
            local_base
            + float(V11_POLICY["alpha"])
            * local_gate[..., None]
            * (consensus - local_base)
        ).clamp_min(0.0)
        desired_slices = amplitude.square()
        desired_slices /= desired_slices.sum(dim=3, keepdim=True).clamp_min(TINY)
        desired_pas = pas_from_slices(desired_slices)
        reference_pdp = pdp_distribution(reference)
        with torch.inference_mode():
            candidate = project_joint(
                reference,
                desired_pas,
                reference_pdp,
                int(V11_POLICY["cycles"]),
                str(V11_POLICY["order"]),
            )
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        if end % 100 == 0 or end == len(v10):
            output.flush()
            print(f"V11 canonical projection {end}/{len(v10)}", flush=True)
    output.flush()
    return validate_channel(V11_CHANNEL)


def score_channel(
    channel: np.ndarray,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]], object]:
    accumulator = OfficialMetricAccumulator()
    for start in range(0, len(channel), batch_size):
        end = min(start + batch_size, len(channel))
        accumulator.add(
            np.asarray(channel[start:end], dtype=np.complex64),
            np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64),
        )
    statistics = accumulator.statistics()
    overall = aggregate_statistics(statistics)
    folds = []
    for fold in range(5):
        result = aggregate_statistics(statistics, fold_id == fold)
        result["fold"] = fold
        folds.append(result)
    return overall, folds, statistics


def load_historical_rows() -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    score_by_version: dict[str, float] = {}
    for version, (path, keys, basis) in HISTORICAL_METRICS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric = canonical_metric(nested_value(payload, keys))
        row = {"version": version, "evaluation_basis": basis, **metric}
        rows.append(row)
        score_by_version[version] = float(metric["score"])
    foundation = json.loads(FOUNDATION_REPORT.read_text(encoding="utf-8"))
    foundation_score = float(foundation["validation"]["base"]["score"])
    score_by_version["Foundation"] = foundation_score
    return rows, score_by_version


def write_csv(rows: list[dict[str, object]]) -> None:
    fields = (
        "version",
        "parent",
        "count",
        "pas",
        "pdp",
        "nmse",
        "nmse_score",
        "score",
        "local_delta_vs_parent",
        "online_score",
        "online_delta_vs_parent",
        "online_minus_local",
        "evaluation_basis",
    )
    with CSV_REPORT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        OOF_META,
        TRAIN_CHANNEL,
        RAW_PATH_PDP,
        POINT_CONTEXT,
        FOLD_IDS,
        E40_PAS,
        V9_ENDPOINT_PAS,
        E39_BASE_PAS,
        E39_SCATTER_PAS,
        FOUNDATION_REPORT,
        *(path for path, _, _ in HISTORICAL_METRICS.values()),
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical replay inputs: {missing}")

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    if len(query_ids) != 500 or len(np.unique(query_ids)) != 500:
        raise ValueError("canonical OOF metadata must contain 500 unique query IDs")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth_power = np.sum(
        np.abs(truth_all[query_ids]) ** 2, axis=(1, 2, 3), dtype=np.float64
    )
    zero_mask = truth_power == 0.0
    if int(zero_mask.sum()) != 31:
        raise ValueError(f"expected 31 zero-channel OOF points, got {zero_mask.sum()}")

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    v9 = build_v9(device, args.batch_size, args.force)
    base_pdp = build_v9_pdp(v9, device, args.batch_size, args.force)
    alpha = predict_v10_alpha(
        query_ids, base_pdp, device, args.batch_size, args.force
    )
    v10 = build_v10(
        v9, query_ids, base_pdp, alpha, device, args.batch_size, args.force
    )
    arrays = pas_sources()
    gate = build_v11_gate(
        v10, arrays, query_ids, device, args.batch_size, args.force
    )
    v11 = build_v11(
        v10, arrays, query_ids, gate, device, args.batch_size, args.force
    )

    replay_metrics: dict[str, dict[str, object]] = {}
    point_payload: dict[str, np.ndarray] = {
        "query_ids": query_ids,
        "fold_id": fold_id,
        "zero_truth_mask": zero_mask,
    }
    for version, channel in (("V9", v9), ("V10", v10), ("V11", v11)):
        metric, folds, statistics = score_channel(
            channel, truth_all, query_ids, fold_id, args.batch_size
        )
        replay_metrics[version] = {
            "overall": metric,
            "folds": folds,
            "nonzero_469_explanatory_only": aggregate_statistics(
                statistics, ~zero_mask
            ),
        }
        point_payload[f"{version.lower()}_pas"] = statistics.pas
        point_payload[f"{version.lower()}_pdp"] = statistics.pdp
        point_payload[f"{version.lower()}_squared_error"] = statistics.squared_error
        point_payload[f"{version.lower()}_truth_energy"] = statistics.truth_energy
        print(f"{version} official 500-point score={metric['score']:.9f}", flush=True)
    np.savez_compressed(POINT_METRICS, **point_payload)

    expected_v9_report = json.loads(
        (
            PATHS.analysis
            / "results"
            / "version_reports"
            / "V9"
            / "V9_joint_sionna_migration.json"
        ).read_text(encoding="utf-8")
    )
    expected_v9 = float(
        expected_v9_report["local_validation"]["selected_policy"]["score"]
    )
    v9_difference = float(replay_metrics["V9"]["overall"]["score"]) - expected_v9
    if not args.skip_historical_control_assertions and abs(v9_difference) > 2e-6:
        raise RuntimeError(
            f"V9 replay control differs from historical score by {v9_difference:+.9g}"
        )

    v10_report = json.loads(
        (
            PATHS.analysis
            / "results"
            / "version_reports"
            / "V10"
            / "V10_full_path_pdp_link_gate.json"
        ).read_text(encoding="utf-8")
    )
    expected_v10_nonzero = float(
        v10_report["local_validation"]["v9_oof_selected"]["score"]
    )
    replayed_v10_nonzero = float(
        replay_metrics["V10"]["nonzero_469_explanatory_only"]["score"]
    )
    v10_nonzero_difference = replayed_v10_nonzero - expected_v10_nonzero
    if (
        not args.skip_historical_control_assertions
        and abs(v10_nonzero_difference) > 2e-6
    ):
        raise RuntimeError(
            "V10 nonzero replay control differs from historical score by "
            f"{v10_nonzero_difference:+.9g}"
        )

    rows, score_by_version = load_historical_rows()
    rows.append(
        {
            "version": "V9",
            "evaluation_basis": "500-point full OOF reconstruction control",
            **replay_metrics["V9"]["overall"],
        }
    )
    rows.append(
        {
            "version": "V10",
            "evaluation_basis": "500-point full OOF replay; cross-fitted link gate",
            **replay_metrics["V10"]["overall"],
        }
    )
    rows.append(
        {
            "version": "V11",
            "evaluation_basis": "500-point full OOF replay; gate reranked on all 500",
            **replay_metrics["V11"]["overall"],
        }
    )
    for version in ("V9", "V10", "V11"):
        score_by_version[version] = float(replay_metrics[version]["overall"]["score"])

    rows.sort(key=lambda row: int(str(row["version"])[1:]))
    for row in rows:
        version = str(row["version"])
        parent = PARENTS[version]
        local_score = float(row["score"])
        online = ONLINE_SCORES[version]
        parent_online = ONLINE_SCORES.get(parent)
        row["parent"] = parent
        row["local_delta_vs_parent"] = local_score - score_by_version[parent]
        row["online_score"] = online
        row["online_delta_vs_parent"] = (
            None if online is None or parent_online is None else online - parent_online
        )
        row["online_minus_local"] = None if online is None else online - local_score

    online_rows = [row for row in rows if row["online_score"] is not None]
    local_absolute = np.asarray(
        [float(row["score"]) for row in online_rows], dtype=np.float64
    )
    online_absolute = np.asarray(
        [float(row["online_score"]) for row in online_rows], dtype=np.float64
    )
    delta_rows = [
        row for row in online_rows if row["online_delta_vs_parent"] is not None
    ]
    local_delta = np.asarray(
        [float(row["local_delta_vs_parent"]) for row in delta_rows],
        dtype=np.float64,
    )
    online_delta = np.asarray(
        [float(row["online_delta_vs_parent"]) for row in delta_rows],
        dtype=np.float64,
    )
    correlation = {
        "absolute": {
            "count": int(len(online_rows)),
            "pearson": float(pearsonr(local_absolute, online_absolute).statistic),
            "spearman": float(spearmanr(local_absolute, online_absolute).statistic),
            "mean_absolute_gap": float(
                np.mean(np.abs(online_absolute - local_absolute))
            ),
            "mean_online_minus_local": float(
                np.mean(online_absolute - local_absolute)
            ),
        },
        "parent_delta": {
            "count": int(len(delta_rows)),
            "pearson": float(pearsonr(local_delta, online_delta).statistic),
            "spearman": float(spearmanr(local_delta, online_delta).statistic),
            "mean_absolute_gap": float(np.mean(np.abs(online_delta - local_delta))),
            "mean_online_minus_local": float(np.mean(online_delta - local_delta)),
        },
    }

    write_csv(rows)
    query_hash = hashlib.sha256(query_ids.tobytes()).hexdigest()
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "evaluation_contract": {
            "query_count": 500,
            "query_ids_sha256": query_hash,
            "zero_truth_count": int(zero_mask.sum()),
            "same_query_set_for_every_version": True,
            "full_channel_prediction_required": True,
            "module_specific_point_filtering": False,
            "metric_implementation": PATHS.relative(
                PATHS.common_code / "official_metrics.py"
            ),
            "subset_metrics_are_absolute_score_evidence": False,
        },
        "policies": {"V10": V10_POLICY, "V11": V11_POLICY},
        "replay_controls": {
            "historical_equality_required": bool(
                not args.skip_historical_control_assertions
            ),
            "v9_full_500": {
                "historical_score": expected_v9,
                "replayed_score": float(
                    replay_metrics["V9"]["overall"]["score"]
                ),
                "difference": v9_difference,
            },
            "v10_nonzero_469_explanatory_only": {
                "historical_score": expected_v10_nonzero,
                "replayed_score": replayed_v10_nonzero,
                "difference": v10_nonzero_difference,
            },
        },
        "local_online_relationship": correlation,
        "versions": rows,
        "replay_details": replay_metrics,
        "artifacts": {
            "v9_channel": PATHS.relative(V9_CHANNEL),
            "v10_channel": PATHS.relative(V10_CHANNEL),
            "v11_channel": PATHS.relative(V11_CHANNEL),
            "point_metrics": PATHS.relative(POINT_METRICS),
            "csv": PATHS.relative(CSV_REPORT),
        },
        "seconds": float(time.time() - started),
    }
    JSON_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {PATHS.relative(JSON_REPORT)}", flush=True)
    print(f"Wrote {PATHS.relative(CSV_REPORT)}", flush=True)


if __name__ == "__main__":
    main()
