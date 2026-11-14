#!/usr/bin/env python3
"""Rebuild the frozen V7 carrier pipeline from sparse-denoised official train H."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from dataclasses import asdict, replace

import numpy as np
import torch
from scipy.spatial import cKDTree

import analyze_v7_complex_carrier_stacking as stacking
import build_v7_complex_carrier_bank as carrier_bank
import run_v7_complex_carrier_exact_projection as v7_projection
from analyze_v7_complex_carrier_stacking import StackConfig
from carrier_calibration import estimate_carrier
from complex_baselines import (
    batched_predict_many_torch,
    predict_many,
    query_site_neighbors,
)
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import synthesize_batch
from run_rbf_spectral_field import OOF_META
from run_v6_regime_component_lifecycle import REGIME_CACHE


VERSION = "Round2-E142-sparse-denoised-v7-carrier"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "sparse_denoised_v7_carrier"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_sparse_denoised_v7_carrier"
COMPONENT_CACHE = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
DENOISED_TRAIN = ARTIFACT_DIR / "official_train_channel_k24_denoised.npy"
DENOISED_POWER = ARTIFACT_DIR / "official_train_channel_k24_power.npy"
OOF_BANK = ARTIFACT_DIR / "oof_denoised_complex_carrier_bank.npy"
TEST_BANK = ARTIFACT_DIR / "test_denoised_complex_carrier_bank.npy"
OOF_DONE = ARTIFACT_DIR / "oof_denoised_complex_carrier_bank_done.npy"
TEST_DONE = ARTIFACT_DIR / "test_denoised_complex_carrier_bank_done.npy"
GRAM_CACHE = ARTIFACT_DIR / "oof_denoised_port_gram.npz"
OOF_STACK = ARTIFACT_DIR / "oof_denoised_stacked_complex_channel.npy"
OOF_PROJECTED = ARTIFACT_DIR / "oof_denoised_stacked_v4_projection.npy"
TEST_STACK = ARTIFACT_DIR / "test_denoised_stacked_complex_channel.npy"
TEST_PROJECTED = ARTIFACT_DIR / "Round2_E142_Denoised_V7_Projection_Test_Channel.npy"
FULL_WEIGHTS = ARTIFACT_DIR / "full_denoised_complex_stack_weights.npz"
DEPLOYMENT = ARTIFACT_DIR / "denoised_complex_carrier_deployment.json"
REPORT = ANALYSIS_DIR / "sparse_denoised_v7_carrier.json"
V7_PROJECTION_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_complex_carrier_stacking"
    / "complex_carrier_exact_projection.json"
)
V7_LIFECYCLE_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_complex_carrier_lifecycle_overlay"
    / "stacked_lifecycle_overlay.json"
)


STACK_CONFIG = StackConfig(
    scope="site",
    ports="port8",
    ridge=1.0,
    sum_penalty=0.0,
)
V7_COMPLETE_SCORE = 0.644456346
V12_COMPLETE_SCORE = 0.677520807
MINIMUM_GAIN_VS_V7 = 0.002


def finite_memmap(path, expected_shape: tuple[int, ...], dtype: np.dtype) -> bool:
    if not path.is_file():
        return False
    value = np.load(path, mmap_mode="r")
    if value.shape != expected_shape or value.dtype != dtype:
        return False
    for start in range(0, len(value), 8):
        block = np.asarray(value[start : start + 8])
        if np.iscomplexobj(block):
            if not np.isfinite(block.real).all() or not np.isfinite(block.imag).all():
                return False
        elif not np.isfinite(block).all():
            return False
    return True


@torch.inference_mode()
def reconstruct_train_channel(
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    source = np.load(TRAIN_CHANNEL, mmap_mode="r")
    expected_shape = tuple(source.shape)
    if finite_memmap(DENOISED_TRAIN, expected_shape, np.dtype(np.complex64)):
        output = np.load(DENOISED_TRAIN, mmap_mode="r")
        reused = True
    else:
        cached = np.load(COMPONENT_CACHE, allow_pickle=False)
        coordinates = cached["coordinates"].astype(np.float32)
        coefficients = cached["coefficients"].astype(np.complex64)
        if coordinates.shape != (len(source), 24, 3):
            raise ValueError(f"unexpected component coordinates: {coordinates.shape}")
        if coefficients.shape != (len(source), 24, 8):
            raise ValueError(f"unexpected component coefficients: {coefficients.shape}")
        output = np.lib.format.open_memmap(
            DENOISED_TRAIN,
            mode="w+",
            dtype=np.complex64,
            shape=expected_shape,
        )
        for start in range(0, len(source), batch_size):
            end = min(start + batch_size, len(source))
            prediction = synthesize_batch(
                coordinates[start:end], coefficients[start:end], device
            )
            output[start:end] = prediction.cpu().numpy().astype(np.complex64)
            output.flush()
            print(f"E142 sparse reconstruction {end}/{len(source)}", flush=True)
        reused = False

    power = np.empty(len(source), dtype=np.float64)
    source_power = np.empty(len(source), dtype=np.float64)
    squared_error = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        clean = np.asarray(output[start:end], dtype=np.complex64)
        noisy = np.asarray(source[start:end], dtype=np.complex64)
        power[start:end] = np.sum(np.abs(clean) ** 2, axis=(1, 2, 3), dtype=np.float64)
        source_power[start:end] = np.sum(
            np.abs(noisy) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        squared_error[start:end] = np.sum(
            np.abs(clean - noisy) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
    np.save(DENOISED_POWER, power)
    nonzero = source_power > 0.0
    relative_residual = squared_error[nonzero] / np.maximum(source_power[nonzero], 1e-30)
    retained_power = power[nonzero] / np.maximum(source_power[nonzero], 1e-30)
    return {
        "path": PATHS.relative(DENOISED_TRAIN),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": True,
        "reused": reused,
        "component_count": 24,
        "official_h_derived": True,
        "nonzero_points": int(np.sum(nonzero)),
        "relative_residual": {
            "mean": float(np.mean(relative_residual)),
            "median": float(np.median(relative_residual)),
            "p90": float(np.quantile(relative_residual, 0.90)),
            "max": float(np.max(relative_residual)),
        },
        "retained_power": {
            "mean": float(np.mean(retained_power)),
            "median": float(np.median(retained_power)),
            "p10": float(np.quantile(retained_power, 0.10)),
            "p90": float(np.quantile(retained_power, 0.90)),
        },
    }


def patch_carrier_paths() -> None:
    carrier_bank.OOF_BANK = OOF_BANK
    carrier_bank.TEST_BANK = TEST_BANK
    stacking.OOF_BANK = OOF_BANK
    stacking.GRAM_CACHE = GRAM_CACHE
    v7_projection.OOF_BANK = OOF_BANK
    v7_projection.TEST_BANK = TEST_BANK
    v7_projection.OOF_STACK = OOF_STACK
    v7_projection.OOF_PROJECTED = OOF_PROJECTED
    v7_projection.TEST_STACK = TEST_STACK
    v7_projection.TEST_PROJECTED = TEST_PROJECTED
    v7_projection.FULL_WEIGHTS = FULL_WEIGHTS
    v7_projection.DEPLOYMENT = DEPLOYMENT


def open_resumable_bank(
    path,
    done_path,
    shape: tuple[int, ...],
) -> tuple[np.memmap, np.memmap]:
    if path.is_file():
        cached = np.load(path, mmap_mode="r+")
        if cached.shape != shape or cached.dtype != np.complex64:
            raise ValueError(f"incompatible existing carrier bank: {path}")
        output = cached
    else:
        output = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.complex64, shape=shape
        )
    if done_path.is_file():
        done = np.load(done_path, mmap_mode="r+")
        if done.shape != (shape[1],) or done.dtype != np.bool_:
            raise ValueError(f"incompatible completion mask: {done_path}")
    else:
        done = np.lib.format.open_memmap(
            done_path, mode="w+", dtype=np.bool_, shape=(shape[1],)
        )
        done[:] = False
        done.flush()
    return output, done


def build_resumable_oof_bank(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    train_site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    configs,
    carrier_pairs: int,
    workers: int,
    device_channel: torch.Tensor | None = None,
    gpu_batch_size: int = 8,
) -> dict[str, object]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    global_to_oof = np.full(len(train_positions), -1, dtype=np.int64)
    global_to_oof[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    shape = (len(configs), len(query_ids), *train_channel.shape[1:])
    output, done = open_resumable_bank(OOF_BANK, OOF_DONE, shape)
    max_k = max(config.k for config in configs)
    carrier_records: dict[str, object] = {}
    generated = 0
    skipped = 0
    pending: list[int] = []

    def flush_pending() -> None:
        if not pending:
            return
        output.flush()
        done[np.asarray(pending, dtype=np.int64)] = True
        done.flush()
        pending.clear()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for fold, fit_indices, fold_queries in carrier_bank._folds_from_cache():
            carrier = estimate_carrier(
                train_channel,
                train_positions,
                train_site,
                bs_positions,
                fit_indices,
                point_power,
                max_pairs_per_site=carrier_pairs,
                seed=20_260_805 + fold * 101,
            )
            carrier_records[str(fold)] = {
                str(site): value.to_dict() for site, value in carrier.items()
            }
            effective_by_site = {
                int(site): [
                    replace(
                        template,
                        beta=value.beta,
                        gamma=value.gamma,
                    )
                    for template in configs
                ]
                for site, value in carrier.items()
            }
            distances, neighbors = query_site_neighbors(
                train_positions,
                train_site,
                fit_indices,
                fold_queries,
                max_k=max_k,
            )
            remaining = []
            for local_row, query_id in enumerate(fold_queries):
                destination = int(global_to_oof[query_id])
                if bool(done[destination]):
                    skipped += 1
                else:
                    remaining.append(local_row)

            def predict_row(local_row: int) -> tuple[int, np.ndarray]:
                query_id = int(fold_queries[local_row])
                destination = int(global_to_oof[query_id])
                donor_ids = neighbors[local_row]
                site = int(train_site[query_id])
                predictions, _ = predict_many(
                    np.asarray(train_channel[donor_ids], dtype=np.complex64),
                    train_positions[donor_ids],
                    train_positions[query_id],
                    distances[local_row],
                    bs_positions[site],
                    effective_by_site[site],
                )
                return destination, predictions

            if device_channel is not None:
                remaining_rows = np.asarray(remaining, dtype=np.int64)
                for start in range(0, len(remaining_rows), gpu_batch_size):
                    local_rows = remaining_rows[start : start + gpu_batch_size]
                    query_ids = fold_queries[local_rows]
                    destinations = global_to_oof[query_ids]
                    prediction, _, _ = batched_predict_many_torch(
                        device_channel,
                        neighbors[local_rows],
                        train_positions[neighbors[local_rows]],
                        train_positions[query_ids],
                        distances[local_rows],
                        bs_positions,
                        train_site[query_ids],
                        effective_by_site,
                        point_power,
                    )
                    output[:, destinations] = (
                        prediction.permute(1, 0, 2, 3, 4).cpu().numpy()
                    )
                    pending.extend(destinations.astype(int).tolist())
                    generated += len(local_rows)
                    flush_pending()
                    print(
                        f"E142 OOF carrier complete={int(done.sum())}/{len(done)} "
                        f"generated_this_run={generated}",
                        flush=True,
                    )
                    del prediction
            else:
                block_size = max(1, workers) * 2
                for start in range(0, len(remaining), block_size):
                    futures = [
                        executor.submit(predict_row, row)
                        for row in remaining[start : start + block_size]
                    ]
                    for future in futures:
                        destination, predictions = future.result()
                        output[:, destination] = predictions
                        pending.append(destination)
                        generated += 1
                        if len(pending) >= 32:
                            flush_pending()
                        if generated % 10 == 0:
                            print(
                                f"E142 OOF carrier complete={int(done.sum()) + len(pending)}/{len(done)} "
                                f"generated_this_run={generated}",
                                flush=True,
                            )
            flush_pending()
    flush_pending()
    if not bool(done.all()):
        raise RuntimeError(f"OOF carrier bank incomplete: {int(done.sum())}/{len(done)}")
    return {
        "generated_this_run": generated,
        "resumed_rows": skipped,
        "completed_rows": int(done.sum()),
        "carrier_by_fold": carrier_records,
    }


def build_resumable_test_bank(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    configs,
    carrier_pairs: int,
    workers: int,
    device_channel: torch.Tensor | None = None,
    gpu_batch_size: int = 8,
) -> dict[str, object]:
    shape = (len(configs), len(test_positions), *train_channel.shape[1:])
    output, done = open_resumable_bank(TEST_BANK, TEST_DONE, shape)
    all_indices = np.arange(len(train_positions), dtype=np.int64)
    carrier = estimate_carrier(
        train_channel,
        train_positions,
        train_site,
        bs_positions,
        all_indices,
        point_power,
        max_pairs_per_site=carrier_pairs,
        seed=20_260_805,
    )
    max_k = max(config.k for config in configs)
    generated = 0
    skipped = 0
    pending: list[int] = []

    def flush_pending() -> None:
        if not pending:
            return
        output.flush()
        done[np.asarray(pending, dtype=np.int64)] = True
        done.flush()
        pending.clear()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for site_value in np.unique(test_site):
            site_value = int(site_value)
            fit = all_indices[train_site == site_value]
            rows = np.flatnonzero(test_site == site_value)
            distances, local = cKDTree(train_positions[fit, :2]).query(
                test_positions[rows, :2], k=max_k, workers=-1
            )
            neighbors = fit[local]
            effective_configs = [
                replace(
                    template,
                    beta=carrier[site_value].beta,
                    gamma=carrier[site_value].gamma,
                )
                for template in configs
            ]
            remaining = []
            for local_row, test_row in enumerate(rows):
                if bool(done[test_row]):
                    skipped += 1
                else:
                    remaining.append(local_row)

            def predict_row(local_row: int) -> tuple[int, np.ndarray]:
                test_row = int(rows[local_row])
                donor_ids = neighbors[local_row]
                predictions, _ = predict_many(
                    np.asarray(train_channel[donor_ids], dtype=np.complex64),
                    train_positions[donor_ids],
                    test_positions[test_row],
                    distances[local_row],
                    bs_positions[site_value],
                    effective_configs,
                )
                return test_row, predictions

            if device_channel is not None:
                configs_by_site = {site_value: effective_configs}
                remaining_rows = np.asarray(remaining, dtype=np.int64)
                for start in range(0, len(remaining_rows), gpu_batch_size):
                    local_rows = remaining_rows[start : start + gpu_batch_size]
                    test_rows = rows[local_rows]
                    prediction, _, _ = batched_predict_many_torch(
                        device_channel,
                        neighbors[local_rows],
                        train_positions[neighbors[local_rows]],
                        test_positions[test_rows],
                        distances[local_rows],
                        bs_positions,
                        test_site[test_rows],
                        configs_by_site,
                        point_power,
                    )
                    output[:, test_rows] = (
                        prediction.permute(1, 0, 2, 3, 4).cpu().numpy()
                    )
                    pending.extend(test_rows.astype(int).tolist())
                    generated += len(local_rows)
                    flush_pending()
                    print(
                        f"E142 test carrier complete={int(done.sum())}/{len(done)} "
                        f"generated_this_run={generated}",
                        flush=True,
                    )
                    del prediction
            else:
                block_size = max(1, workers) * 2
                for start in range(0, len(remaining), block_size):
                    futures = [
                        executor.submit(predict_row, row)
                        for row in remaining[start : start + block_size]
                    ]
                    for future in futures:
                        test_row, predictions = future.result()
                        output[:, test_row] = predictions
                        pending.append(test_row)
                        generated += 1
                        if len(pending) >= 32:
                            flush_pending()
                        if generated % 10 == 0:
                            print(
                                f"E142 test carrier complete={int(done.sum()) + len(pending)}/{len(done)} "
                                f"generated_this_run={generated}",
                                flush=True,
                            )
            flush_pending()
    flush_pending()
    if not bool(done.all()):
        raise RuntimeError(f"test carrier bank incomplete: {int(done.sum())}/{len(done)}")
    return {
        "generated_this_run": generated,
        "resumed_rows": skipped,
        "completed_rows": int(done.sum()),
        "carrier": {str(site): value.to_dict() for site, value in carrier.items()},
    }


def build_bank(
    phase: str,
    carrier_pairs: int,
    workers: int,
    device: torch.device,
    gpu_batch_size: int,
) -> dict[str, object]:
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    train_channel = np.load(DENOISED_TRAIN, mmap_mode="r")
    point_power = np.load(DENOISED_POWER, allow_pickle=False).astype(np.float64)
    configs = carrier_bank.carrier_configs()
    device_channel = None
    if device.type == "cuda":
        device_channel = torch.as_tensor(
            np.asarray(train_channel), dtype=torch.complex64, device=device
        )
    details: dict[str, object] = {}
    if phase in ("oof", "all"):
        generation = build_resumable_oof_bank(
            train_channel,
            train_positions,
            train_site,
            setup.bs_positions,
            point_power,
            configs,
            carrier_pairs,
            workers,
            device_channel,
            gpu_batch_size,
        )
        details["oof_generation"] = generation
        details["oof"] = carrier_bank.validate_bank(
            OOF_BANK, (len(configs), 500, 256, 4, 192)
        )
    if phase in ("test", "all"):
        generation = build_resumable_test_bank(
            train_channel,
            train_positions,
            test_positions,
            train_site,
            test_site,
            setup.bs_positions,
            point_power,
            configs,
            carrier_pairs,
            workers,
            device_channel,
            gpu_batch_size,
        )
        details["test_generation"] = generation
        details["test"] = carrier_bank.validate_bank(
            TEST_BANK, (len(configs), len(test_positions), 256, 4, 192)
        )
    return {
        "config_names": [config.name for config in configs],
        "details": details,
    }


def metric_delta(
    candidate: dict[str, float | int], reference: dict[str, float | int]
) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(reference[key])
        for key in ("pas", "pdp", "nmse", "nmse_score", "score")
    }


def evaluate_oof(device: torch.device, batch_size: int) -> dict[str, object]:
    cache = stacking.compute_gram(device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)

    stack_detail = v7_projection.build_oof_stack(
        cache, STACK_CONFIG, site, regime, batch_size
    )
    projection_detail = v7_projection.project_v4_policy(
        device, batch_size, query_ids, site
    )
    truth = np.load(TRAIN_CHANNEL, mmap_mode="r")
    raw = v7_projection.score_channel(
        np.load(OOF_STACK, mmap_mode="r"),
        truth,
        query_ids,
        site,
        fold_id,
        batch_size,
    )
    projected = v7_projection.score_channel(
        np.load(OOF_PROJECTED, mmap_mode="r"),
        truth,
        query_ids,
        site,
        fold_id,
        batch_size,
    )

    reference_report = json.loads(V7_PROJECTION_REPORT.read_text(encoding="utf-8"))
    reference = reference_report["projected_score"]["nested"]
    candidate = projected["nested"]
    fold_deltas = []
    for current, old in zip(candidate["folds"], reference["folds"], strict=True):
        if int(current["fold"]) != int(old["fold"]):
            raise RuntimeError("V7 fold ordering changed")
        fold_deltas.append(
            {
                "fold": int(current["fold"]),
                **metric_delta(current, old),
            }
        )
    overall = candidate["overall"]
    positive_folds = int(sum(row["score"] > 0.0 for row in fold_deltas))
    minimum_gate = (
        float(overall["score"]) >= V7_COMPLETE_SCORE + MINIMUM_GAIN_VS_V7
        and positive_folds >= 4
        and float(overall["pas"]) >= float(reference["overall"]["pas"])
        and float(overall["pdp"]) >= float(reference["overall"]["pdp"])
        and float(overall["nmse"]) <= float(reference["overall"]["nmse"])
    )
    return {
        "stack_config": {
            "name": STACK_CONFIG.name,
            **asdict(STACK_CONFIG),
        },
        "stack_artifact": stack_detail,
        "projection_artifact": projection_detail,
        "raw_score": raw,
        "projected_score": projected,
        "v7_exact_projection_reference": reference,
        "delta_vs_v7_exact_projection": metric_delta(
            overall, reference["overall"]
        ),
        "fold_deltas_vs_v7_exact_projection": fold_deltas,
        "positive_folds": positive_folds,
        "deployment_gate": {
            "minimum_gain_vs_complete_v7": MINIMUM_GAIN_VS_V7,
            "complete_v7_score": V7_COMPLETE_SCORE,
            "required_minimum_score": V7_COMPLETE_SCORE + MINIMUM_GAIN_VS_V7,
            "required_positive_folds": 4,
            "pas_non_worse": float(overall["pas"])
            >= float(reference["overall"]["pas"]),
            "pdp_non_worse": float(overall["pdp"])
            >= float(reference["overall"]["pdp"]),
            "nmse_non_worse": float(overall["nmse"])
            <= float(reference["overall"]["nmse"]),
            "minimum_gate_passed": minimum_gate,
            "v12_score": V12_COMPLETE_SCORE,
            "final_goal_passed": float(overall["score"]) > V12_COMPLETE_SCORE,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("reconstruct", "oof", "test", "all"),
        default="oof",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--carrier-pairs", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    patch_carrier_paths()
    device = torch.device(args.device)
    reconstruction = reconstruct_train_channel(device, args.batch_size)
    payload: dict[str, object] = {
        "experiment": VERSION,
        "status": "running",
        "official_inputs_only": True,
        "sionna_used": False,
        "primary_hypothesis": (
            "The residual outside the 24-component off-grid delay-angle model is dominated "
            "by official complex AWGN; removing it before rebuilding the otherwise frozen "
            "V7 carrier bank should improve PAS, PDP, and coherent NMSE together."
        ),
        "reconstruction": reconstruction,
    }
    if args.phase in ("oof", "test", "all"):
        bank_phase = "all" if args.phase == "all" else args.phase
        payload["carrier_bank"] = build_bank(
            bank_phase,
            args.carrier_pairs,
            args.workers,
            device,
            args.gpu_batch_size,
        )
    if args.phase in ("oof", "all"):
        payload["evaluation"] = evaluate_oof(device, args.batch_size)
        passed = bool(payload["evaluation"]["deployment_gate"]["minimum_gate_passed"])
        payload["status"] = "oof_gate_passed" if passed else "analysis_only_gate_failed"
    elif args.phase == "test":
        payload["status"] = "test_bank_ready"
    else:
        payload["status"] = "reconstruction_ready"
    payload["seconds"] = float(time.time() - started)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
