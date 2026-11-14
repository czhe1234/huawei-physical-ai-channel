#!/usr/bin/env python3
"""Evaluate clean Sionna path pretraining on held-out official H rows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import analyze_v205_sionna_config_calibration as calibration
import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v451_official_path_representation_probe as v451
import analyze_v452_v445_representation_over_strong_anchor as v452
import generate_v218_official_multispectrum_physics_ai as v218
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import train_v445_candidate_path_token_model as v445
from project_paths import PATHS


VERSION_NUMBER = 560
CACHE = PATHS.cache_dir(VERSION_NUMBER)
OFFICIAL_CACHE = CACHE / "official"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / "v560_geometry_only_pretraining_transfer.json"
TARGET_SOURCE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
GEOMETRY_SOURCE = PATHS.shared_features / "height_geometry_cache.npz"
LEGACY_FULL_CHECKPOINT = (
    PATHS.checkpoints / "v445_candidate_path_token_model" / "component_route_best.pt"
)
PILOT_CHECKPOINT_ROOT = PATHS.checkpoints / "v560_geometry_only_pretraining_pilot"
OFFICIAL_POSITION_SOURCE = PATHS.cache_dir(475) / "positions.npz"
OFFICIAL_TOKEN_ROOT = PATHS.cache_dir(477)
OFFICIAL_FEATURE_SOURCE = OFFICIAL_TOKEN_ROOT / "v477_path_candidate_feature.npy"
OFFICIAL_VALID_SOURCE = OFFICIAL_TOKEN_ROOT / "v477_path_candidate_valid.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=56_101)
    parser.add_argument("--full-h-neighbors", type=int, default=8)
    parser.add_argument("--skip-full-h", action="store_true")
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def input_view(max_paths: int) -> dict[str, np.ndarray]:
    manifest = np.load(OFFICIAL_POSITION_SOURCE)
    return {
        "positions": manifest["positions"].astype(np.float32),
        "path_candidate_feature": np.load(
            OFFICIAL_FEATURE_SOURCE, mmap_mode="r"
        )[:, :max_paths],
        "path_candidate_valid": np.load(
            OFFICIAL_VALID_SOURCE, mmap_mode="r"
        )[:, :max_paths],
    }


def checkpoint_path(name: str) -> Path:
    if name == "legacy_full_k128":
        return LEGACY_FULL_CHECKPOINT
    return PILOT_CHECKPOINT_ROOT / name / "component_route_best.pt"


def load_model(path: Path, device: torch.device) -> v445.CandidatePathTokenModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = v445.CandidatePathTokenModel(coordinate_dropout=0.0)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def cached_embedding(
    name: str,
    model: v445.CandidatePathTokenModel,
    inputs: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    path = OFFICIAL_CACHE / f"embedding_{name}.npy"
    if path.is_file():
        return np.load(path)
    ids = np.arange(len(inputs["positions"]), dtype=np.int64)
    value = v451.path_embedding(model, inputs, ids, device, batch_size)
    np.save(path, value)
    return value


def load_embeddings(
    args: argparse.Namespace, device: torch.device
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    inputs128 = input_view(128)
    raw_path = OFFICIAL_CACHE / "embedding_raw_geometry_k128.npy"
    if raw_path.is_file():
        raw = np.load(raw_path)
    else:
        raw = v451.raw_geometry_features(inputs128)
        np.save(raw_path, raw)
    methods: dict[str, np.ndarray] = {"raw_geometry": raw.astype(np.float32)}

    torch.manual_seed(args.seed)
    random_model = v445.CandidatePathTokenModel(coordinate_dropout=0.0).to(device)
    methods["random_encoder"] = cached_embedding(
        "random_k128", random_model, inputs128, device, args.batch_size
    )
    del random_model
    torch.cuda.empty_cache()

    checkpoints = {
        "legacy_full_k128": (checkpoint_path("legacy_full_k128"), inputs128),
        "legacy_pilot_k128": (checkpoint_path("legacy_k128"), inputs128),
        "clean_pilot_k128": (checkpoint_path("clean_k128"), inputs128),
        "clean_finetune_k128": (checkpoint_path("clean_finetune_k128"), inputs128),
    }
    for name, (path, inputs) in checkpoints.items():
        require(path)
        model = load_model(path, device)
        methods[name] = cached_embedding(
            name, model, inputs, device, args.batch_size
        ).astype(np.float32)
        del model
        torch.cuda.empty_cache()
    return methods, inputs128


def random_folds(valid_ids: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [part.astype(np.int64) for part in np.array_split(rng.permutation(valid_ids), 5)]


def protocol_probe(
    name: str,
    folds: list[np.ndarray],
    methods: dict[str, np.ndarray],
    valid_ids: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
    seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray], np.ndarray]:
    targets = v452.build_fold_targets(
        folds, valid_ids, positions, blocked, truth_sh, seed
    )
    results: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    base = None
    for method_name, feature in methods.items():
        result, current_base, prediction = v452.probe_method(
            f"{name}_{method_name}",
            feature[:2000],
            targets,
            valid_ids,
            truth_sh,
            truth_d,
        )
        results[method_name] = result
        predictions[method_name] = prediction
        if base is None:
            base = current_base
        elif not np.allclose(base[valid_ids], current_base[valid_ids], atol=1e-7):
            raise ValueError("Fold anchor changed across representation controls")

    rng = np.random.default_rng(seed + 91)
    shuffled = methods["clean_finetune_k128"].copy()
    shuffled[valid_ids] = shuffled[rng.permutation(valid_ids)]
    result, current_base, prediction = v452.probe_method(
        f"{name}_clean_candidate_shuffle",
        shuffled[:2000],
        targets,
        valid_ids,
        truth_sh,
        truth_d,
    )
    results["clean_candidate_shuffle"] = result
    predictions["clean_candidate_shuffle"] = prediction
    assert base is not None
    return results, predictions, base


def build_full_h_anchor(
    folds: list[np.ndarray],
    valid_ids: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    neighbors: int,
) -> Path:
    output_path = OFFICIAL_CACHE / f"random_oof_anchor_h_k{neighbors}.npy"
    if output_path.is_file():
        value = np.load(output_path, mmap_mode="r")
        if value.shape == (2000, 256, 4, 192) and value.dtype == np.complex64:
            return output_path
        raise ValueError(f"Invalid cached full-H anchor {value.shape}/{value.dtype}")
    source_ids = np.full((2000, neighbors), -1, dtype=np.int64)
    source_weights = np.zeros((2000, neighbors), dtype=np.float32)
    for query_ids in folds:
        bank_ids = np.setdiff1d(valid_ids, query_ids)
        index, weight = v218.neighbor_index_weights(
            positions[query_ids],
            blocked[query_ids],
            positions[bank_ids],
            blocked[bank_ids],
            k=v218.ANCHOR_K,
        )
        index = index[:, :neighbors]
        weight = weight[:, :neighbors]
        weight /= np.maximum(np.sum(weight, axis=1, keepdims=True), 1e-30)
        source_ids[query_ids] = bank_ids[index]
        source_weights[query_ids] = weight

    truth = np.load(TRAIN_H, mmap_mode="r")
    partial = output_path.with_name(output_path.stem + ".partial.npy")
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.complex64, shape=truth.shape
    )
    output[:] = 0
    for start in range(0, len(valid_ids), 4):
        rows = valid_ids[start : start + 4]
        for row in rows:
            values = np.asarray(truth[source_ids[row]], dtype=np.complex64)
            output[row] = np.sum(
                values * source_weights[row, :, None, None, None],
                axis=0,
                dtype=np.complex64,
            )
        if start % 200 == 0:
            output.flush()
            print(f"v560 full-H anchor {min(start + 4, len(valid_ids))}/{len(valid_ids)}", flush=True)
    output.flush()
    del output
    partial.replace(output_path)
    return output_path


def full_h_metrics(
    anchor_path: Path,
    joint: np.ndarray | None,
    valid_ids: np.ndarray,
) -> dict[str, float]:
    anchor = np.load(anchor_path, mmap_mode="r")
    truth = np.load(TRAIN_H, mmap_mode="r")
    pas_parts = []
    pdp_parts = []
    error = 0.0
    energy = 0.0
    for start in range(0, len(valid_ids), 2):
        ids = valid_ids[start : start + 2]
        base_h = np.asarray(anchor[ids], dtype=np.complex64)
        if joint is None:
            prediction = base_h
        else:
            base_support, _ = v423.leaked_support(base_h)
            desired, _, _ = v435.build_desired_support(
                base_support,
                joint[ids],
                np.ones(len(ids), dtype=np.float32),
            )
            prediction = v425.project_full_support(base_h, desired)
            base_power = np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
            pred_power = np.sum(np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64)
            scale = np.sqrt(base_power / np.maximum(pred_power, 1e-30)).astype(np.float32)
            prediction *= scale[:, None, None, None]
        target = np.asarray(truth[ids], dtype=np.complex64)
        scores = calibration.spectral_scores(prediction, calibration.spectral_views(target))
        pas_parts.append(scores["pas"])
        pdp_parts.append(scores["pdp"])
        error += float(np.sum(np.abs(prediction - target) ** 2, dtype=np.float64))
        energy += float(np.sum(np.abs(target) ** 2, dtype=np.float64))
    nmse = error / max(energy, 1e-30)
    pas = float(np.mean(np.concatenate(pas_parts)))
    pdp = float(np.mean(np.concatenate(pdp_parts)))
    nmse_score = 1.0 / (1.0 + nmse)
    return {
        "pas": pas,
        "pdp": pdp,
        "nmse": float(nmse),
        "nmse_score": float(nmse_score),
        "total": float(0.4 * pas + 0.4 * pdp + 0.2 * nmse_score),
    }


def metric_delta(model: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {key: float(model[key] - base[key]) for key in model}


def main() -> None:
    args = parse_args()
    required = (
        OFFICIAL_POSITION_SOURCE,
        OFFICIAL_FEATURE_SOURCE,
        OFFICIAL_VALID_SOURCE,
        TARGET_SOURCE,
        TRAIN_H,
        GEOMETRY_SOURCE,
        checkpoint_path("legacy_k128"),
        checkpoint_path("clean_k128"),
        checkpoint_path("clean_finetune_k128"),
        LEGACY_FULL_CHECKPOINT,
    )
    for path in required:
        require(path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"v560 requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )
    OFFICIAL_CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device("cuda")
    methods, inputs = load_embeddings(args, device)
    target = np.load(TARGET_SOURCE)
    truth_sh = target["sh"].astype(np.float32)
    truth_d = target["d"].astype(np.float32)
    path_count = np.sum(
        np.asarray(inputs["path_candidate_valid"][:2000], dtype=np.int16), axis=1
    )
    complete_candidate = path_count < 128
    valid_ids = np.flatnonzero(
        target["valid"].astype(bool) & complete_candidate
    ).astype(np.int64)
    positions = np.asarray(inputs["positions"][:2000, :2], dtype=np.float64)
    blocked = np.load(GEOMETRY_SOURCE)["train_blocked"].astype(np.float32)
    random = random_folds(valid_ids, args.seed)
    spatial = v401.confirmatory_folds(valid_ids, positions)

    random_result, random_prediction, random_base = protocol_probe(
        "random80_20",
        random,
        methods,
        valid_ids,
        positions,
        blocked,
        truth_sh,
        truth_d,
        args.seed,
    )
    spatial_result, spatial_prediction, spatial_base = protocol_probe(
        "spatial_stress",
        spatial,
        methods,
        valid_ids,
        positions,
        blocked,
        truth_sh,
        truth_d,
        args.seed + 10_000,
    )
    np.savez_compressed(
        OFFICIAL_CACHE / "v560_official_transfer_oof.npz",
        valid_ids=valid_ids,
        random_base=random_base[valid_ids].astype(np.float16),
        spatial_base=spatial_base[valid_ids].astype(np.float16),
        **{
            f"random_{name}": value[valid_ids].astype(np.float16)
            for name, value in random_prediction.items()
        },
        **{
            f"spatial_{name}": value[valid_ids].astype(np.float16)
            for name, value in spatial_prediction.items()
        },
    )

    full_h: dict[str, object] | None = None
    if not args.skip_full_h:
        anchor_path = build_full_h_anchor(
            random, valid_ids, positions, blocked, args.full_h_neighbors
        )
        base_metric = full_h_metrics(anchor_path, None, valid_ids)
        full_h = {"base": base_metric, "methods": {}}
        for name in (
            "random_encoder",
            "legacy_full_k128",
            "legacy_pilot_k128",
            "clean_pilot_k128",
            "clean_finetune_k128",
            "clean_candidate_shuffle",
        ):
            metric = full_h_metrics(anchor_path, random_prediction[name], valid_ids)
            full_h["methods"][name] = {
                "metric": metric,
                "delta": metric_delta(metric, base_metric),
            }

    clean_random = random_result["clean_finetune_k128"]
    clean_spatial = spatial_result["clean_finetune_k128"]
    controls = ("legacy_full_k128", "random_encoder", "clean_candidate_shuffle")
    clean_gain = float(clean_random["delta"]["weighted_52_sh_48_d"])
    control_gain = max(
        float(random_result[name]["delta"]["weighted_52_sh_48_d"])
        for name in controls
    )
    full_h_positive = bool(
        full_h is None
        or full_h["methods"]["clean_finetune_k128"]["delta"]["total"] > 0.0
    )
    passed = bool(
        clean_gain > control_gain + 5e-4
        and clean_random["positive_folds"] >= 4
        and clean_spatial["delta"]["weighted_52_sh_48_d"] > 0.0
        and clean_spatial["positive_folds"] >= 3
        and full_h_positive
    )
    report = {
        "method": "v560_geometry_only_pretraining_transfer",
        "status": "pilot_passed_expand_pretraining" if passed else "pilot_failed_no_expansion",
        "primary_hypothesis": (
            "Removing power-selected candidate membership should make synthetic path "
            "pretraining transfer better than matched legacy, random, and shuffled controls."
        ),
        "candidate_completeness": {
            "official_valid_rows": int(np.sum(target["valid"].astype(bool))),
            "saturated_rows_excluded": int(
                np.sum(target["valid"].astype(bool) & ~complete_candidate)
            ),
            "evaluated_complete_rows": int(len(valid_ids)),
            "rule": "path_count<128; saturated rows use fallback in any future deployment",
        },
        "protocols": {
            "random80_20": random_result,
            "spatial_stress": spatial_result,
        },
        "full_h_random80_20": full_h,
        "decision": {
            "criterion": (
                "clean random gain > every matched control by 0.0005 with >=4/5 positive "
                "folds, positive spatial gain with >=3/5 folds, and positive full-H total"
            ),
            "clean_random_gain": clean_gain,
            "strongest_control_gain": control_gain,
            "passed": passed,
            "next_step": (
                "expand clean geometry-only pretraining beyond 20k"
                if passed
                else "stop top-K cleanup as the main hypothesis and redesign path supervision"
            ),
        },
        "single_gpu": {
            "logical_device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0),
        },
        "seconds": float(time.time() - started),
        "outputs": {
            "oof": PATHS.relative(OFFICIAL_CACHE / "v560_official_transfer_oof.npz"),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
