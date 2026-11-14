#!/usr/bin/env python3
"""Materialize strict Clean E183 actions for one isolated hole family."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

import analyze_e119_v12_unique_path_complement as e119
import analyze_e121_standalone_unique_path_carrier as e121
import analyze_e122_strict_complex_stack_unique_path as e122
import analyze_v7_complex_carrier_stacking as stacking
import build_v7_complex_carrier_bank as carrier_bank
import run_e142_denoised_v7_carrier as e142
import run_v4_calibrated_exact_projection as v4
import run_v7_complex_carrier_exact_projection as v7_projection
import train_e115_complete_path_state_operator as e115
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from analyze_v7_stacked_lifecycle_overlay import OUTPUT as V7_OUTPUT
from analyze_v7_stacked_lifecycle_overlay import REPORT as V7_LIFECYCLE_REPORT
from carrier_calibration import estimate_carrier
from complex_baselines import predict_one
from data import load_positions
from project_paths import PATHS
from run_v6_regime_component_lifecycle import REGIME_CACHE


VERSION = "Round2-E183-clean-family-actions"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e183_family_actions"
REPORT = (
    PATHS.analysis
    / "results"
    / "round2_e183_strict_action_labels"
    / "family_action_generation.json"
)
ENDPOINT = (
    PATHS.artifacts
    / "round2"
    / "fivefold_unique_path_v12"
    / "test_fivefold_mass_coordinate_endpoint.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
ACTION_NAMES = (
    "V7",
    "V4",
    "E121_path",
    "E122_path",
    "E142_raw",
    "E142_projected",
    "carrier_00_nearest",
    "carrier_07_raw_idw_k24",
    "carrier_08_radial_idw_k24",
    "carrier_11_radial_affine_k24_r0.1",
    "carrier_16_affine_k24_steer_axis90_v8pi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--carrier-pairs", type=int, default=500)
    parser.add_argument("--fit-samples-per-site", type=int, default=320)
    parser.add_argument("--gram-batch-size", type=int, default=4)
    parser.add_argument("--gpu-batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def output_path(name: str) -> Path:
    return ARTIFACT_DIR / f"{name}.npy"


def valid_channel(path: Path, count: int) -> bool:
    if not path.is_file():
        return False
    value = np.load(path, mmap_mode="r")
    if value.shape != (count, 256, 4, 192) or value.dtype != np.complex64:
        return False
    for start in range(0, count, 8):
        block = np.asarray(value[start : start + 8])
        if not np.isfinite(block.real).all() or not np.isfinite(block.imag).all():
            return False
    return True


def save_channel(name: str, value: np.ndarray, force: bool) -> dict[str, object]:
    path = output_path(name)
    count = len(value)
    if valid_channel(path, count) and not force:
        return {
            "path": PATHS.relative(path),
            "shape": list(np.load(path, mmap_mode="r").shape),
            "dtype": "complex64",
            "finite": True,
            "reused": True,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.complex64, shape=value.shape
    )
    for start in range(0, count, 8):
        end = min(start + 8, count)
        output[start:end] = np.asarray(value[start:end], dtype=np.complex64)
    output.flush()
    if not valid_channel(path, count):
        raise RuntimeError(f"invalid E183 action output: {name}")
    return {
        "path": PATHS.relative(path),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": True,
        "reused": False,
    }


def query_table(
    fit: np.ndarray,
    query_ids: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.empty((len(query_ids), k), dtype=np.float64)
    neighbors = np.empty((len(query_ids), k), dtype=np.int64)
    for site_value in np.unique(site[query_ids]):
        rows = np.flatnonzero(site[query_ids] == site_value)
        bank = fit[site[fit] == site_value]
        local_distance, local = cKDTree(positions[bank, :2]).query(
            positions[query_ids[rows], :2], k=k, workers=-1
        )
        distances[rows] = np.asarray(local_distance, dtype=np.float64)
        neighbors[rows] = bank[np.asarray(local, dtype=np.int64)]
    return distances, neighbors


def endpoint_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    endpoint = np.load(ENDPOINT, allow_pickle=False)
    return e119.expand_endpoint(
        endpoint["pas"].astype(np.float32),
        endpoint["pdp"].astype(np.float32),
        device,
    )


def apply_endpoint(
    base_np: np.ndarray,
    endpoint: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.empty_like(np.asarray(base_np), dtype=np.complex64)
    for start in range(0, len(base_np), batch_size):
        end = min(start + batch_size, len(base_np))
        base = torch.as_tensor(
            np.asarray(base_np[start:end], dtype=np.complex64), device=device
        )
        prediction = e119.make_candidate(
            base,
            pas_distribution(base),
            pdp_distribution(base),
            endpoint[0][start:end],
            endpoint[1][start:end],
            e121.PRIMARY_ACTION,
        )
        output[start:end] = prediction.cpu().numpy().astype(np.complex64)
    return output


def build_e121(
    device: torch.device,
    batch_size: int,
    carrier_pairs: int,
    force: bool,
) -> dict[str, object]:
    if valid_channel(output_path("E121_path"), len(load_positions()[2])) and not force:
        return save_channel(
            "E121_path", np.load(output_path("E121_path"), mmap_mode="r"), False
        )
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    positions = np.concatenate((train_positions, test_positions), axis=0)
    site = np.concatenate((train_site, test_site), axis=0)
    train_count = len(train_positions)
    fit = np.arange(train_count, dtype=np.int64)
    query_ids = train_count + np.arange(len(test_positions), dtype=np.int64)
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    point_power = np.load(e115.COMPONENTS, allow_pickle=False)["point_power"].astype(
        np.float64
    )
    carrier = estimate_carrier(
        train_h,
        train_positions,
        train_site,
        setup.bs_positions,
        fit,
        point_power,
        max_pairs_per_site=carrier_pairs,
        seed=20_260_808,
    )
    template = e121.radial_idw_config()
    distance, neighbors = query_table(fit, query_ids, positions, site, template.k)
    base = np.empty((len(test_positions), 256, 4, 192), dtype=np.complex64)
    for row, query_id in enumerate(query_ids):
        site_value = int(site[query_id])
        config = replace(
            template,
            beta=carrier[site_value].beta,
            gamma=carrier[site_value].gamma,
        )
        base[row], _ = predict_one(
            np.asarray(train_h[neighbors[row]], dtype=np.complex64),
            positions[neighbors[row]],
            positions[query_id],
            distance[row],
            setup.bs_positions[site_value],
            config,
        )
    prediction = apply_endpoint(base, endpoint_tensors(device), device, batch_size)
    result = save_channel("E121_path", prediction, force)
    result["carrier"] = {str(key): value.to_dict() for key, value in carrier.items()}
    return result


def build_e122(
    device: torch.device,
    batch_size: int,
    carrier_pairs: int,
    samples_per_site: int,
    gram_batch_size: int,
    force: bool,
) -> dict[str, object]:
    if valid_channel(output_path("E122_path"), len(load_positions()[2])) and not force:
        return save_channel(
            "E122_path", np.load(output_path("E122_path"), mmap_mode="r"), False
        )
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    positions = np.concatenate((train_positions, test_positions), axis=0)
    site = np.concatenate((train_site, test_site), axis=0)
    train_count = len(train_positions)
    fit = np.arange(train_count, dtype=np.int64)
    query_ids = train_count + np.arange(len(test_positions), dtype=np.int64)
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    point_power = np.load(e115.COMPONENTS, allow_pickle=False)["point_power"].astype(
        np.float64
    )
    templates = carrier_bank.carrier_configs()
    carrier = estimate_carrier(
        train_h,
        train_positions,
        train_site,
        setup.bs_positions,
        fit,
        point_power,
        max_pairs_per_site=carrier_pairs,
        seed=20_260_808,
    )
    weights, fit_record = e122.fit_fold_weights(
        -1,
        fit,
        templates,
        carrier,
        train_h,
        positions,
        site,
        setup.bs_positions,
        samples_per_site,
        gram_batch_size,
        device,
    )
    distance, neighbors = e122.neighbor_table(
        fit, query_ids, positions, site, max(config.k for config in templates)
    )
    base = np.empty((len(test_positions), 256, 4, 192), dtype=np.complex64)
    for row, query_id in enumerate(query_ids):
        candidates = e122.generate_carriers(
            int(query_id),
            neighbors[row],
            distance[row],
            templates,
            carrier,
            train_h,
            positions,
            site,
            setup.bs_positions,
        )
        base[row] = e122.combine_carriers(candidates, weights[int(site[query_id])])
    prediction = apply_endpoint(base, endpoint_tensors(device), device, batch_size)
    result = save_channel("E122_path", prediction, force)
    result["fit"] = fit_record
    result["weight_norm"] = {
        str(key): float(np.linalg.norm(value)) for key, value in weights.items()
    }
    return result


def build_e121_e122_gpu(
    device: torch.device,
    batch_size: int,
    gpu_batch_size: int,
    carrier_pairs: int,
    samples_per_site: int,
    gram_batch_size: int,
    force: bool,
) -> dict[str, dict[str, object]]:
    _, _, test_positions, _, _ = load_positions()
    count = len(test_positions)
    if (
        valid_channel(output_path("E121_path"), count)
        and valid_channel(output_path("E122_path"), count)
        and not force
    ):
        return {
            "E121_path": save_channel(
                "E121_path", np.load(output_path("E121_path"), mmap_mode="r"), False
            ),
            "E122_path": save_channel(
                "E122_path", np.load(output_path("E122_path"), mmap_mode="r"), False
            ),
        }

    setup, train_positions, test_positions, train_site, test_site = load_positions()
    positions = np.concatenate((train_positions, test_positions), axis=0)
    site = np.concatenate((train_site, test_site), axis=0)
    train_count = len(train_positions)
    fit = np.arange(train_count, dtype=np.int64)
    query_ids = train_count + np.arange(count, dtype=np.int64)
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    point_power = np.load(e115.COMPONENTS, allow_pickle=False)["point_power"].astype(
        np.float64
    )
    templates = carrier_bank.carrier_configs()
    carrier = estimate_carrier(
        train_h,
        train_positions,
        train_site,
        setup.bs_positions,
        fit,
        point_power,
        max_pairs_per_site=carrier_pairs,
        seed=20_260_808,
    )
    weights, fit_record = e122.fit_fold_weights(
        -1,
        fit,
        templates,
        carrier,
        train_h,
        positions,
        site,
        setup.bs_positions,
        samples_per_site,
        gram_batch_size,
        device,
    )

    e121_base = np.empty((count, 256, 4, 192), dtype=np.complex64)
    e122_base = np.empty_like(e121_base)
    device_channel = torch.as_tensor(
        np.asarray(train_h, dtype=np.complex64), device=device
    )
    max_k = max(config.k for config in templates)
    for site_value in np.unique(test_site):
        site_value = int(site_value)
        bank = fit[train_site == site_value]
        rows = np.flatnonzero(test_site == site_value)
        distance, local = cKDTree(train_positions[bank, :2]).query(
            test_positions[rows, :2], k=max_k, workers=-1
        )
        neighbors = bank[np.asarray(local, dtype=np.int64)]
        configs_by_site = {
            site_value: [
                replace(
                    template,
                    beta=carrier[site_value].beta,
                    gamma=carrier[site_value].gamma,
                )
                for template in templates
            ]
        }
        for start in range(0, len(rows), gpu_batch_size):
            end = min(start + gpu_batch_size, len(rows))
            test_rows = rows[start:end]
            prediction, _, _ = carrier_bank.batched_predict_many_torch(
                device_channel,
                neighbors[start:end],
                train_positions[neighbors[start:end]],
                test_positions[test_rows],
                np.asarray(distance[start:end], dtype=np.float64),
                setup.bs_positions,
                test_site[test_rows],
                configs_by_site,
                point_power,
            )
            candidate = prediction.cpu().numpy().astype(np.complex64, copy=False)
            e121_base[test_rows] = candidate[:, 8]
            for local_row, test_row in enumerate(test_rows):
                e122_base[test_row] = e122.combine_carriers(
                    candidate[local_row], weights[site_value]
                )
            print(
                f"E121/E122 GPU carriers site={site_value} {end}/{len(rows)}",
                flush=True,
            )
            del prediction, candidate
    del device_channel

    endpoint = endpoint_tensors(device)
    e121_prediction = apply_endpoint(e121_base, endpoint, device, batch_size)
    e122_prediction = apply_endpoint(e122_base, endpoint, device, batch_size)
    e121_result = save_channel("E121_path", e121_prediction, force)
    e121_result["carrier"] = {
        str(key): value.to_dict() for key, value in carrier.items()
    }
    e121_result["gpu_batched"] = True
    e122_result = save_channel("E122_path", e122_prediction, force)
    e122_result["fit"] = fit_record
    e122_result["weight_norm"] = {
        str(key): float(np.linalg.norm(value)) for key, value in weights.items()
    }
    e122_result["gpu_batched"] = True
    return {"E121_path": e121_result, "E122_path": e122_result}


def score_scales(report: dict[str, object], kind: str) -> dict[int, float]:
    decisions = report["evaluation"][f"{kind}_score"]["full_oof"][
        "scale_decisions"
    ]
    return {int(row["site"]): float(row["scale"]) for row in decisions}


def build_e142(
    device: torch.device, batch_size: int, force: bool
) -> dict[str, dict[str, object]]:
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    e142.patch_carrier_paths()
    metadata = np.load(e119.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    oof_site = train_site[query_ids].astype(np.int8)
    oof_regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(
        np.int8
    )
    cache = stacking.compute_gram(device)
    v7_projection.build_test_stack(
        cache,
        e142.STACK_CONFIG,
        oof_site,
        oof_regime,
        test_site,
        batch_size,
    )
    report = json.loads(e142.REPORT.read_text(encoding="utf-8"))
    raw_scale = score_scales(report, "raw")
    raw = np.asarray(np.load(e142.TEST_STACK, mmap_mode="r"), dtype=np.complex64)
    raw = np.asarray(
        raw
        * np.asarray([raw_scale[int(value)] for value in test_site], dtype=np.float32)[
            :, None, None, None
        ],
        dtype=np.complex64,
    )
    raw_result = save_channel("E142_raw", raw, force)
    projected_scale = score_scales(report, "projected")
    v7_projection.project_test_v4_policy(
        device, batch_size, test_site, projected_scale
    )
    projected_result = save_channel(
        "E142_projected",
        np.load(e142.TEST_PROJECTED, mmap_mode="r"),
        force,
    )
    return {"raw": raw_result, "projected": projected_result}


def link_existing_actions(force: bool) -> dict[str, dict[str, object]]:
    _, _, test_positions, _, _ = load_positions()
    count = len(test_positions)
    def file_record(path) -> dict[str, object]:
        if not valid_channel(path, count):
            raise RuntimeError(f"invalid existing E183 action: {path}")
        value = np.load(path, mmap_mode="r")
        return {
            "provider": "file",
            "path": PATHS.relative(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": True,
            "reused": True,
        }

    def v7_record() -> dict[str, object]:
        if valid_channel(V7_OUTPUT, count):
            return file_record(V7_OUTPUT)
        if not V7_LIFECYCLE_REPORT.is_file():
            raise RuntimeError(
                f"missing V7 lifecycle output and report: {V7_OUTPUT}"
            )
        lifecycle = json.loads(V7_LIFECYCLE_REPORT.read_text(encoding="utf-8"))
        if lifecycle.get("selection", {}).get("winner") is not None:
            raise RuntimeError(
                "V7 lifecycle selected a winner but its output is missing or invalid"
            )
        record = file_record(v7_projection.TEST_PROJECTED)
        record["fallback"] = "lifecycle_selection_no_winner"
        record["lifecycle_report"] = PATHS.relative(V7_LIFECYCLE_REPORT)
        return record

    output = {
        "V7": v7_record(),
        "V4": file_record(v4.OUTPUT),
    }
    bank = np.load(carrier_bank.TEST_BANK, mmap_mode="r")
    carrier_actions = {
        "carrier_00_nearest": 0,
        "carrier_07_raw_idw_k24": 7,
        "carrier_08_radial_idw_k24": 8,
        "carrier_11_radial_affine_k24_r0.1": 11,
        "carrier_16_affine_k24_steer_axis90_v8pi": 16,
    }
    if bank.shape[1] != count:
        raise RuntimeError(f"carrier query count mismatch: {bank.shape[1]} != {count}")
    for name, index in carrier_actions.items():
        block = np.asarray(bank[index])
        if block.dtype != np.complex64 or not np.isfinite(block).all():
            raise RuntimeError(f"invalid carrier action: {name}")
        output[name] = {
            "provider": "carrier",
            "path": PATHS.relative(carrier_bank.TEST_BANK),
            "carrier_index": index,
            "shape": list(block.shape),
            "dtype": str(block.dtype),
            "finite": True,
            "reused": True,
        }
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if not ENDPOINT.is_file():
        raise FileNotFoundError(f"missing Clean E120 endpoint: {ENDPOINT}")
    actions = link_existing_actions(args.force)
    actions.update(build_e121_e122_gpu(
        device,
        args.batch_size,
        args.gpu_batch_size,
        args.carrier_pairs,
        args.fit_samples_per_site,
        args.gram_batch_size,
        args.force,
    ))
    e142_actions = build_e142(device, args.batch_size, args.force)
    actions["E142_raw"] = e142_actions["raw"]
    actions["E142_projected"] = e142_actions["projected"]
    missing = [name for name in ACTION_NAMES if name not in actions]
    if missing:
        raise RuntimeError(f"missing E183 actions: {missing}")
    report = {
        "experiment": VERSION,
        "status": "eleven_clean_actions_ready",
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "hidden_truth_read": False,
        "actions": {name: actions[name] for name in ACTION_NAMES},
        "seconds": float(time.time() - started),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
