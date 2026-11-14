#!/usr/bin/env python3
"""Benchmark and build the pre-Sionna Round 2 complex-channel foundation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from complex_baselines import (
    BaselineConfig,
    batched_predict_many_torch,
    predict_many,
    predict_one,
    query_site_neighbors,
)
from carrier_calibration import CarrierEstimate, estimate_carrier
from data import load_positions
from official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics,
    point_metric_statistics_torch,
)
from project_paths import PATHS
from spatial_hole_cv import SpatialHoleFold, build_spatial_hole_folds


ANALYSIS_DIR = PATHS.analysis / "results" / "round2_foundation_benchmark"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "foundation"
REPORT = ANALYSIS_DIR / "round2_foundation_benchmark.json"
POINT_METRICS = ANALYSIS_DIR / "round2_foundation_point_metrics.npz"
SPLITS = ARTIFACT_DIR / "round2_spatial_hole_folds.npz"
POWER_CACHE = ARTIFACT_DIR / "round2_train_point_power.npy"
OUTPUT = ARTIFACT_DIR / "Round2_Foundation_Test_Channel.npy"
SELECTION = ARTIFACT_DIR / "round2_foundation_selection.json"


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def load_or_compute_power(train_channel: np.ndarray, batch_size: int) -> np.ndarray:
    if POWER_CACHE.is_file():
        cached = np.load(POWER_CACHE)
        if cached.shape == (len(train_channel),):
            return cached.astype(np.float64)
    power = np.empty(len(train_channel), dtype=np.float64)
    for start in range(0, len(train_channel), batch_size):
        end = min(start + batch_size, len(train_channel))
        channel = np.asarray(train_channel[start:end], dtype=np.complex64)
        power[start:end] = np.sum(
            np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        print(f"point power {end}/{len(train_channel)}", flush=True)
    np.save(POWER_CACHE, power)
    return power


def config_templates() -> list[BaselineConfig]:
    configs = [BaselineConfig(name="nearest", method="nearest", k=1)]
    for k in (4, 8, 16, 24):
        configs.append(BaselineConfig(name=f"raw_idw_k{k}", method="raw_idw", k=k))
        configs.append(BaselineConfig(name=f"radial_idw_k{k}", method="radial_idw", k=k))
    for k in (8, 16, 24, 32):
        for ridge in (0.10, 0.30, 1.00):
            configs.append(
                BaselineConfig(
                    name=f"radial_affine_k{k}_r{ridge:g}",
                    method="radial_affine",
                    k=k,
                    ridge=ridge,
                )
            )
    for k in (8, 16, 24):
        configs.append(
            BaselineConfig(
                name=f"radial_idw_k{k}_linear_power",
                method="radial_idw",
                k=k,
                power_mode="linear",
            )
        )
        configs.append(
            BaselineConfig(
                name=f"radial_idw_k{k}_log_power",
                method="radial_idw",
                k=k,
                power_mode="log_nonzero",
            )
        )
    for threshold in (0.70, 0.85):
        configs.append(
            BaselineConfig(
                name=f"radial_idw_k16_zero{threshold:.2f}",
                method="radial_idw",
                k=16,
                zero_threshold=threshold,
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
    names = [config.name for config in configs]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate baseline names")
    return configs


def save_folds(folds: list[SpatialHoleFold]) -> None:
    payload: dict[str, np.ndarray] = {}
    for fold in folds:
        payload[f"fold{fold.fold}_fit"] = fold.fit_indices.astype(np.int32)
        payload[f"fold{fold.fold}_query"] = fold.query_indices.astype(np.int32)
        payload[f"fold{fold.fold}_ignored"] = fold.ignored_indices.astype(np.int32)
    np.savez_compressed(SPLITS, **payload)


def _metric_store(configs: list[BaselineConfig], count: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        config.name: {
            "pas": np.full(count, np.nan, dtype=np.float64),
            "pdp": np.full(count, np.nan, dtype=np.float64),
            "error": np.full(count, np.nan, dtype=np.float64),
            "energy": np.full(count, np.nan, dtype=np.float64),
            "fold": np.full(count, -1, dtype=np.int8),
            "zero_probability": np.full(count, np.nan, dtype=np.float32),
            "zero_gate": np.zeros(count, dtype=bool),
        }
        for config in configs
    }


def _statistics(values: dict[str, np.ndarray], ids: np.ndarray) -> PointMetricStatistics:
    return PointMetricStatistics(
        pas=values["pas"][ids],
        pdp=values["pdp"][ids],
        squared_error=values["error"][ids],
        truth_energy=values["energy"][ids],
    )


def evaluate_folds(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    train_site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    folds: list[SpatialHoleFold],
    configs: list[BaselineConfig],
    carrier_pairs: int,
    workers: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[int, dict[int, CarrierEstimate]]]:
    metrics = _metric_store(configs, len(train_positions))
    carrier_by_fold: dict[int, dict[int, CarrierEstimate]] = {}
    max_k = max(config.k for config in configs)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for fold in folds:
            carrier = estimate_carrier(
                train_channel,
                train_positions,
                train_site,
                bs_positions,
                fold.fit_indices,
                point_power,
                max_pairs_per_site=carrier_pairs,
                seed=20_260_805 + fold.fold * 101,
            )
            carrier_by_fold[fold.fold] = carrier
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
                fold.fit_indices,
                fold.query_indices,
                max_k=max_k,
            )

            def evaluate_row(row: int):
                query_index = int(fold.query_indices[row])
                neighbor_index = neighbors[row]
                site = int(train_site[query_index])
                prediction_batch, diagnostics = predict_many(
                    np.asarray(train_channel[neighbor_index], dtype=np.complex64),
                    train_positions[neighbor_index],
                    train_positions[query_index],
                    distances[row],
                    bs_positions[site],
                    effective_by_site[site],
                )
                return query_index, diagnostics, prediction_batch

            block_size = max(1, workers) * 2
            completed = 0
            for start in range(0, len(fold.query_indices), block_size):
                futures = [
                    executor.submit(evaluate_row, row)
                    for row in range(
                        start, min(start + block_size, len(fold.query_indices))
                    )
                ]
                results = [future.result() for future in futures]
                query_indices = np.asarray(
                    [value[0] for value in results], dtype=np.int64
                )
                prediction_batch = np.stack([value[2] for value in results])
                truth_batch = np.asarray(
                    train_channel[query_indices], dtype=np.complex64
                )
                flat_prediction = prediction_batch.reshape(
                    -1, *prediction_batch.shape[2:]
                )
                if device.type == "cuda":
                    prediction_tensor = torch.as_tensor(
                        flat_prediction, device=device
                    )
                    truth_tensor = torch.as_tensor(truth_batch, device=device)
                    truth_tensor = (
                        truth_tensor[:, None]
                        .expand(-1, len(configs), -1, -1, -1)
                        .reshape_as(prediction_tensor)
                    )
                    with torch.inference_mode():
                        scores = point_metric_statistics_torch(
                            prediction_tensor, truth_tensor
                        )
                else:
                    expanded_truth = np.broadcast_to(
                        truth_batch[:, None], prediction_batch.shape
                    ).reshape(flat_prediction.shape)
                    scores = point_metric_statistics(
                        flat_prediction, expanded_truth
                    )
                score_pas = scores.pas.reshape(len(results), len(configs))
                score_pdp = scores.pdp.reshape(len(results), len(configs))
                score_error = scores.squared_error.reshape(
                    len(results), len(configs)
                )
                score_energy = scores.truth_energy.reshape(
                    len(results), len(configs)
                )
                for local_result, (query_index, diagnostics, _) in enumerate(results):
                    for index, config in enumerate(configs):
                        values = metrics[config.name]
                        values["pas"][query_index] = score_pas[local_result, index]
                        values["pdp"][query_index] = score_pdp[local_result, index]
                        values["error"][query_index] = score_error[local_result, index]
                        values["energy"][query_index] = score_energy[local_result, index]
                        values["fold"][query_index] = fold.fold
                        values["zero_probability"][query_index] = diagnostics[index]["zero_probability"]
                        values["zero_gate"][query_index] = diagnostics[index]["zero_gate"]
                    completed += 1
                    if completed % 10 == 0 or completed == len(fold.query_indices):
                        print(
                            f"foundation fold={fold.fold} queries={completed}/{len(fold.query_indices)}",
                            flush=True,
                        )
    return metrics, carrier_by_fold


@torch.inference_mode()
def evaluate_folds_gpu(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    train_site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    folds: list[SpatialHoleFold],
    configs: list[BaselineConfig],
    carrier_pairs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[int, dict[int, CarrierEstimate]]]:
    metrics = _metric_store(configs, len(train_positions))
    carrier_by_fold: dict[int, dict[int, CarrierEstimate]] = {}
    max_k = max(config.k for config in configs)
    device_channel = torch.as_tensor(
        np.asarray(train_channel), dtype=torch.complex64, device=device
    )

    for fold in folds:
        carrier = estimate_carrier(
            train_channel,
            train_positions,
            train_site,
            bs_positions,
            fold.fit_indices,
            point_power,
            max_pairs_per_site=carrier_pairs,
            seed=20_260_805 + fold.fold * 101,
        )
        carrier_by_fold[fold.fold] = carrier
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
            fold.fit_indices,
            fold.query_indices,
            max_k=max_k,
        )
        for start in range(0, len(fold.query_indices), batch_size):
            end = min(start + batch_size, len(fold.query_indices))
            query_ids = fold.query_indices[start:end]
            prediction, zero_probability, zero_gate = batched_predict_many_torch(
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
            truth = device_channel[
                torch.as_tensor(query_ids, dtype=torch.long, device=device)
            ]
            truth = (
                truth[:, None]
                .expand(-1, len(configs), -1, -1, -1)
                .reshape(-1, *truth.shape[1:])
            )
            scores = point_metric_statistics_torch(
                prediction.reshape(-1, *prediction.shape[2:]), truth
            )
            local_count = len(query_ids)
            score_pas = scores.pas.reshape(local_count, len(configs))
            score_pdp = scores.pdp.reshape(local_count, len(configs))
            score_error = scores.squared_error.reshape(local_count, len(configs))
            score_energy = scores.truth_energy.reshape(local_count, len(configs))
            for row, query_index in enumerate(query_ids):
                for index, config in enumerate(configs):
                    values = metrics[config.name]
                    values["pas"][query_index] = score_pas[row, index]
                    values["pdp"][query_index] = score_pdp[row, index]
                    values["error"][query_index] = score_error[row, index]
                    values["energy"][query_index] = score_energy[row, index]
                    values["fold"][query_index] = fold.fold
                    values["zero_probability"][query_index] = zero_probability[
                        row, index
                    ]
                    values["zero_gate"][query_index] = zero_gate[row, index]
            print(
                f"foundation fold={fold.fold} queries={end}/{len(fold.query_indices)}",
                flush=True,
            )
            del prediction, truth
    return metrics, carrier_by_fold


def build_report_rows(
    metrics: dict[str, dict[str, np.ndarray]],
    configs: list[BaselineConfig],
    folds: list[SpatialHoleFold],
    train_site: np.ndarray,
    point_power: np.ndarray,
) -> list[dict[str, object]]:
    query_ids = np.sort(np.concatenate([fold.query_indices for fold in folds]))
    rows: list[dict[str, object]] = []
    for config in configs:
        values = metrics[config.name]
        row: dict[str, object] = {
            "name": config.name,
            "config": config.to_dict(),
            "all": aggregate_statistics(_statistics(values, query_ids)),
            "zero_gate_count": int(np.sum(values["zero_gate"][query_ids])),
            "folds": [],
            "sites": {},
            "truth_zero": aggregate_statistics(
                _statistics(values, query_ids[point_power[query_ids] <= 1e-30])
            ),
            "truth_nonzero": aggregate_statistics(
                _statistics(values, query_ids[point_power[query_ids] > 1e-30])
            ),
        }
        for fold in folds:
            row["folds"].append(
                {
                    "fold": fold.fold,
                    **aggregate_statistics(_statistics(values, fold.query_indices)),
                }
            )
        for site in np.unique(train_site):
            ids = query_ids[train_site[query_ids] == site]
            row["sites"][str(int(site))] = aggregate_statistics(_statistics(values, ids))
        rows.append(row)
    rows.sort(key=lambda row: float(row["all"]["score"]), reverse=True)
    return rows


def _find_config(configs: list[BaselineConfig], name: str) -> BaselineConfig:
    for config in configs:
        if config.name == name:
            return config
    raise KeyError(name)


def selected_oof_summary(
    metrics: dict[str, dict[str, np.ndarray]],
    folds: list[SpatialHoleFold],
    train_site: np.ndarray,
    site_config_names: dict[int, str],
) -> dict[str, object]:
    query_ids = np.sort(np.concatenate([fold.query_indices for fold in folds]))
    combined = {
        "pas": np.full(len(train_site), np.nan, dtype=np.float64),
        "pdp": np.full(len(train_site), np.nan, dtype=np.float64),
        "error": np.full(len(train_site), np.nan, dtype=np.float64),
        "energy": np.full(len(train_site), np.nan, dtype=np.float64),
    }
    for site, name in site_config_names.items():
        ids = query_ids[train_site[query_ids] == site]
        for key in combined:
            combined[key][ids] = metrics[name][key][ids]

    def aggregate(ids: np.ndarray) -> dict[str, float | int]:
        return aggregate_statistics(
            PointMetricStatistics(
                pas=combined["pas"][ids],
                pdp=combined["pdp"][ids],
                squared_error=combined["error"][ids],
                truth_energy=combined["energy"][ids],
            )
        )

    return {
        "all": aggregate(query_ids),
        "folds": [
            {"fold": fold.fold, **aggregate(fold.query_indices)} for fold in folds
        ],
        "sites": {
            str(int(site)): aggregate(query_ids[train_site[query_ids] == site])
            for site in np.unique(train_site)
        },
    }


def generate_foundation(
    train_channel: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    configs: list[BaselineConfig],
    rows: list[dict[str, object]],
    carrier_pairs: int,
) -> dict[str, object]:
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
    site_config_names = {
        int(site): max(rows, key=lambda row: float(row["sites"][str(int(site))]["score"]))["name"]
        for site in np.unique(train_site)
    }
    output = np.lib.format.open_memmap(
        OUTPUT,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_positions),) + tuple(train_channel.shape[1:]),
    )
    diagnostics = defaultdict(list)
    for site in np.unique(test_site):
        site = int(site)
        fit_indices = all_indices[train_site == site]
        query_rows = np.flatnonzero(test_site == site)
        config = replace(
            _find_config(configs, str(site_config_names[site])),
            beta=carrier[site].beta,
            gamma=carrier[site].gamma,
        )
        tree = cKDTree(train_positions[fit_indices, :2])
        distances, local = tree.query(
            test_positions[query_rows, :2], k=config.k, workers=-1
        )
        if config.k == 1:
            distances = distances[:, None]
            local = local[:, None]
        neighbors = fit_indices[local]
        for local_row, output_row in enumerate(query_rows):
            neighbor_index = neighbors[local_row]
            prediction, diagnostic = predict_one(
                np.asarray(train_channel[neighbor_index], dtype=np.complex64),
                train_positions[neighbor_index],
                test_positions[output_row],
                distances[local_row],
                bs_positions[site],
                config,
            )
            output[output_row] = prediction
            diagnostics[f"site{site}_zero_probability"].append(diagnostic["zero_probability"])
            diagnostics[f"site{site}_zero_gate"].append(diagnostic["zero_gate"])
            if (local_row + 1) % 25 == 0 or local_row + 1 == len(query_rows):
                output.flush()
                print(
                    f"foundation test site={site} generated={local_row + 1}/{len(query_rows)}",
                    flush=True,
                )
    output.flush()
    finite = bool(np.isfinite(output).all())
    if not finite:
        raise ValueError("foundation output contains non-finite values")
    return {
        "output": PATHS.relative(OUTPUT),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "site_configs": {str(key): value for key, value in site_config_names.items()},
        "carrier": {str(site): value.to_dict() for site, value in carrier.items()},
        "test_diagnostics": {
            key: summarize(np.asarray(value, dtype=np.float64))
            if "probability" in key
            else int(np.sum(value))
            for key, value in diagnostics.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-batch", type=int, default=16)
    parser.add_argument("--carrier-pairs", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--gpu-batch-size", type=int, default=8)
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    train_channel = np.load(
        PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    point_power = load_or_compute_power(train_channel, args.power_batch)
    folds = build_spatial_hole_folds(
        train_positions, test_positions, train_site, test_site
    )
    save_folds(folds)
    configs = config_templates()
    device = torch.device(args.device)
    if device.type == "cuda":
        metrics, fold_carrier = evaluate_folds_gpu(
            train_channel,
            train_positions,
            train_site,
            setup.bs_positions,
            point_power,
            folds,
            configs,
            args.carrier_pairs,
            args.gpu_batch_size,
            device,
        )
    else:
        metrics, fold_carrier = evaluate_folds(
            train_channel,
            train_positions,
            train_site,
            setup.bs_positions,
            point_power,
            folds,
            configs,
            args.carrier_pairs,
            args.workers,
            device,
        )
    rows = build_report_rows(metrics, configs, folds, train_site, point_power)

    npz_payload: dict[str, np.ndarray] = {"point_power": point_power}
    for name, values in metrics.items():
        for key, value in values.items():
            npz_payload[f"{name}__{key}"] = value
    np.savez_compressed(POINT_METRICS, **npz_payload)

    generation = None
    if not args.skip_generate:
        generation = generate_foundation(
            train_channel,
            train_positions,
            test_positions,
            train_site,
            test_site,
            setup.bs_positions,
            point_power,
            configs,
            rows,
            args.carrier_pairs,
        )
        selected_names = {
            int(site): str(name) for site, name in generation["site_configs"].items()
        }
        generation["selected_oof"] = selected_oof_summary(
            metrics, folds, train_site, selected_names
        )
        SELECTION.write_text(
            json.dumps(generation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    report = {
        "analysis": "round2_foundation_benchmark",
        "status": "analysis_only_pre_sionna_foundation",
        "primary_hypothesis": (
            "A site-aware radial phase-aligned local model should preserve complex path "
            "structure across Round 2 spatial holes better than raw nearest-neighbor or IDW."
        ),
        "inputs": {
            "train_channel": "datasets/round2/Round2_Train_Channel.npy",
            "train_positions": "datasets/round2/Round2_Train_Pos.npy",
            "test_positions_for_hole_geometry_only": "datasets/round2/Round2_Test_Pos.npy",
            "sionna_used": False,
            "round1_prediction_or_cache_used": False,
        },
        "metric": {
            "formula": "0.4*PAS + 0.4*PDP + 0.2/(1+global_NMSE)",
            "pas": "FFT over [16H,8V], 2 polarizations retained in the 256-vector",
            "pdp": "IFFT over 192 subcarriers per 256x4 link",
            "zero_norm_rule": "denominator=1e-12 only when the norm product is exactly zero",
        },
        "validation": {
            "fold_count": len(folds),
            "query_count": int(sum(len(fold.query_indices) for fold in folds)),
            "folds": [
                {
                    "fold": fold.fold,
                    "query_count": int(len(fold.query_indices)),
                    "fit_count": int(len(fold.fit_indices)),
                    "ignored_count": int(len(fold.ignored_indices)),
                    "sites": list(fold.rows),
                }
                for fold in folds
            ],
        },
        "carrier_fit_only_by_fold": {
            str(fold): {
                str(site): estimate.to_dict() for site, estimate in site_values.items()
            }
            for fold, site_values in fold_carrier.items()
        },
        "carrier_parameter_semantics": (
            "The beta/gamma values stored in each config template are defaults only. "
            "Every validation prediction replaces them with that fold's fit-only, "
            "site-specific estimate; final generation uses the all-train estimate."
        ),
        "config_count": len(configs),
        "results": rows,
        "selected_generation": generation,
        "outputs": {
            "report": PATHS.relative(REPORT),
            "point_metrics": PATHS.relative(POINT_METRICS),
            "folds": PATHS.relative(SPLITS),
            "selection": PATHS.relative(SELECTION) if generation else None,
            "foundation_channel": PATHS.relative(OUTPUT) if generation else None,
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"top_results": rows[:10], "generation": generation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
