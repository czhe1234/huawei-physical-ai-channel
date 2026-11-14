#!/usr/bin/env python3
"""Build a diverse official-data complex carrier bank for Round 2 V7."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from dataclasses import replace

import numpy as np
import torch
from scipy.spatial import cKDTree

from carrier_calibration import estimate_carrier
from complex_baselines import (
    BaselineConfig,
    batched_predict_many_torch,
    predict_many,
    query_site_neighbors,
)
from data import load_positions
from project_paths import PATHS
from run_foundation_benchmark import POWER_CACHE, load_or_compute_power
from run_rbf_spectral_field import FOLD_CACHE, OOF_META


VERSION = "Round2-E44-diverse-complex-carrier-bank"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "complex_carrier_bank"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_complex_carrier_bank"
OOF_BANK = ARTIFACT_DIR / "oof_complex_carrier_bank.npy"
TEST_BANK = ARTIFACT_DIR / "test_complex_carrier_bank.npy"
METADATA = ARTIFACT_DIR / "complex_carrier_bank_metadata.json"
REPORT = ANALYSIS_DIR / "complex_carrier_bank.json"


def carrier_configs() -> list[BaselineConfig]:
    configs = [BaselineConfig(name="nearest", method="nearest", k=1)]
    for k in (4, 8, 16, 24):
        configs.append(BaselineConfig(name=f"raw_idw_k{k}", method="raw_idw", k=k))
        configs.append(
            BaselineConfig(name=f"radial_idw_k{k}", method="radial_idw", k=k)
        )
    for k, ridge in ((8, 0.10), (16, 1.00), (24, 0.10), (32, 0.10)):
        configs.append(
            BaselineConfig(
                name=f"radial_affine_k{k}_r{ridge:g}",
                method="radial_affine",
                k=k,
                ridge=ridge,
            )
        )
    for axis in (0.0, 90.0):
        for vertical_scale, tag in ((1.5 * np.pi, "v1p5pi"), (8.0 * np.pi, "v8pi")):
            configs.append(
                BaselineConfig(
                    name=f"affine_k24_steer_axis{int(axis)}_{tag}",
                    method="radial_affine",
                    k=24,
                    ridge=0.30,
                    steering_alpha=1.0,
                    horizontal_axis_deg=axis,
                    vertical_scale=vertical_scale,
                )
            )
    if len({config.name for config in configs}) != len(configs):
        raise RuntimeError("duplicate carrier names")
    return configs


def _folds_from_cache() -> list[tuple[int, np.ndarray, np.ndarray]]:
    cached = np.load(FOLD_CACHE, allow_pickle=False)
    return [
        (
            fold,
            cached[f"fold{fold}_fit"].astype(np.int64),
            cached[f"fold{fold}_query"].astype(np.int64),
        )
        for fold in range(5)
    ]


def build_oof_bank(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    train_site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    configs: list[BaselineConfig],
    carrier_pairs: int,
    workers: int,
    device_channel: torch.Tensor | None = None,
    gpu_batch_size: int = 8,
) -> tuple[np.ndarray, dict[str, object]]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    global_to_oof = np.full(len(train_positions), -1, dtype=np.int64)
    global_to_oof[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    expected = (len(configs), len(query_ids), *train_channel.shape[1:])
    if OOF_BANK.is_file():
        cached = np.load(OOF_BANK, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.complex64:
            return cached, {"reused": True}
    output = np.lib.format.open_memmap(
        OOF_BANK, mode="w+", dtype=np.complex64, shape=expected
    )
    max_k = max(config.k for config in configs)
    carrier_records: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for fold, fit_indices, fold_queries in _folds_from_cache():
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

            def predict_row(local_row: int) -> tuple[int, np.ndarray]:
                query_id = int(fold_queries[local_row])
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
                return int(global_to_oof[query_id]), predictions

            completed = 0
            if device_channel is not None:
                for start in range(0, len(fold_queries), gpu_batch_size):
                    end = min(start + gpu_batch_size, len(fold_queries))
                    query_ids = fold_queries[start:end]
                    prediction, _, _ = batched_predict_many_torch(
                        device_channel,
                        neighbors[start:end],
                        train_positions[neighbors[start:end]],
                        train_positions[query_ids],
                        distances[start:end],
                        bs_positions,
                        train_site[query_ids],
                        effective_by_site,
                        point_power,
                    )
                    destinations = global_to_oof[query_ids]
                    output[:, destinations] = (
                        prediction.permute(1, 0, 2, 3, 4).cpu().numpy()
                    )
                    completed = end
                    print(
                        f"carrier bank OOF fold={fold} {completed}/{len(fold_queries)}",
                        flush=True,
                    )
                    del prediction
            else:
                block_size = max(1, workers) * 2
                for start in range(0, len(fold_queries), block_size):
                    futures = [
                        executor.submit(predict_row, row)
                        for row in range(start, min(start + block_size, len(fold_queries)))
                    ]
                    for future in futures:
                        destination, predictions = future.result()
                        output[:, destination] = predictions
                        completed += 1
                        if completed % 10 == 0 or completed == len(fold_queries):
                            print(
                                f"carrier bank OOF fold={fold} {completed}/{len(fold_queries)}",
                                flush=True,
                            )
            output.flush()
    output.flush()
    return np.load(OOF_BANK, mmap_mode="r"), {
        "reused": False,
        "carrier_by_fold": carrier_records,
    }


def build_test_bank(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    configs: list[BaselineConfig],
    carrier_pairs: int,
    workers: int,
    device_channel: torch.Tensor | None = None,
    gpu_batch_size: int = 8,
) -> tuple[np.ndarray, dict[str, object]]:
    expected = (len(configs), len(test_positions), *train_channel.shape[1:])
    if TEST_BANK.is_file():
        cached = np.load(TEST_BANK, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.complex64:
            return cached, {"reused": True}
    output = np.lib.format.open_memmap(
        TEST_BANK, mode="w+", dtype=np.complex64, shape=expected
    )
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
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for site_value in np.unique(test_site):
            site_value = int(site_value)
            fit = all_indices[train_site == site_value]
            rows = np.flatnonzero(test_site == site_value)
            tree = cKDTree(train_positions[fit, :2])
            distances, local = tree.query(
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

            completed = 0
            if device_channel is not None:
                configs_by_site = {site_value: effective_configs}
                for start in range(0, len(rows), gpu_batch_size):
                    end = min(start + gpu_batch_size, len(rows))
                    test_rows = rows[start:end]
                    prediction, _, _ = batched_predict_many_torch(
                        device_channel,
                        neighbors[start:end],
                        train_positions[neighbors[start:end]],
                        test_positions[test_rows],
                        distances[start:end],
                        bs_positions,
                        test_site[test_rows],
                        configs_by_site,
                        point_power,
                    )
                    output[:, test_rows] = (
                        prediction.permute(1, 0, 2, 3, 4).cpu().numpy()
                    )
                    completed = end
                    print(
                        f"carrier bank test site={site_value} {completed}/{len(rows)}",
                        flush=True,
                    )
                    del prediction
            else:
                block_size = max(1, workers) * 2
                for start in range(0, len(rows), block_size):
                    futures = [
                        executor.submit(predict_row, row)
                        for row in range(start, min(start + block_size, len(rows)))
                    ]
                    for future in futures:
                        test_row, predictions = future.result()
                        output[:, test_row] = predictions
                        completed += 1
                        if completed % 10 == 0 or completed == len(rows):
                            print(
                                f"carrier bank test site={site_value} {completed}/{len(rows)}",
                                flush=True,
                            )
            output.flush()
    output.flush()
    return np.load(TEST_BANK, mmap_mode="r"), {
        "reused": False,
        "carrier": {str(site): value.to_dict() for site, value in carrier.items()},
    }


def validate_bank(path, expected_shape: tuple[int, ...]) -> dict[str, object]:
    value = np.load(path, mmap_mode="r")
    if value.shape != expected_shape:
        raise ValueError(f"bank shape {value.shape} != {expected_shape}")
    finite = True
    for config in range(len(value)):
        for start in range(0, value.shape[1], 4):
            block = np.asarray(value[config, start : start + 4])
            finite = finite and bool(
                np.isfinite(block.real).all() and np.isfinite(block.imag).all()
            )
    if not finite:
        raise ValueError(f"non-finite carrier bank: {path}")
    return {
        "path": PATHS.relative(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": finite,
        "bytes": int(path.stat().st_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "test", "all"), default="oof")
    parser.add_argument("--carrier-pairs", type=int, default=500)
    parser.add_argument("--power-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--gpu-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    train_channel = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    point_power = load_or_compute_power(train_channel, args.power_batch)
    configs = carrier_configs()
    device = torch.device(args.device)
    device_channel = None
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        device_channel = torch.as_tensor(
            np.asarray(train_channel), dtype=torch.complex64, device=device
        )
    details: dict[str, object] = {}
    if args.phase in ("oof", "all"):
        _, details["oof_generation"] = build_oof_bank(
            train_channel,
            train_positions,
            train_site,
            setup.bs_positions,
            point_power,
            configs,
            args.carrier_pairs,
            args.workers,
            device_channel,
            args.gpu_batch_size,
        )
        details["oof"] = validate_bank(
            OOF_BANK, (len(configs), 500, 256, 4, 192)
        )
    if args.phase in ("test", "all"):
        _, details["test_generation"] = build_test_bank(
            train_channel,
            train_positions,
            test_positions,
            train_site,
            test_site,
            setup.bs_positions,
            point_power,
            configs,
            args.carrier_pairs,
            args.workers,
            device_channel,
            args.gpu_batch_size,
        )
        details["test"] = validate_bank(
            TEST_BANK, (len(configs), len(test_positions), 256, 4, 192)
        )
    metadata = {
        "experiment": VERSION,
        "official_inputs_only": True,
        "sionna_used": False,
        "config_names": [config.name for config in configs],
        "configs": [config.to_dict() for config in configs],
        "details": details,
        "seconds": float(time.time() - started),
    }
    text = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    METADATA.write_text(text, encoding="utf-8")
    REPORT.write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
