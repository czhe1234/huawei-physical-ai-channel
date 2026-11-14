#!/usr/bin/env python3
"""Build Round 2 V4 from predictable grouped spectra and exact projections."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict

import numpy as np
import torch

from analyze_v4_exact_joint_projection import (
    combine_with_v3,
    continue_trajectory,
    initialize_trajectory,
    metric_components,
    select_nested,
    summarize_candidates,
    v3_nested_baseline,
)
from analyze_v4_predictability_calibrated_groups import (
    GROUPED_PAS,
    GROUPED_PDP,
    PAS_COMPONENTS,
    PDP_COMPONENTS,
    blend_group_pas,
    blend_group_pdp,
    encode,
    globally_shrunk_targets,
    normalize_group_amplitude,
    pca_basis,
    predict_grouped,
)
from build_v3_anchored_neural_rbf import (
    TEST_WEIGHTS,
    TRAIN_AD,
    TRAIN_PAS,
    TRAIN_PDP,
    spectral_target,
)
from data import load_positions
from evaluate_v3_anchored_projection import (
    TARGET_AD,
    TARGET_PAS,
    TARGET_PDP,
    target_batch,
)
from pre_sionna_spectral import project_channel
from project_paths import PATHS
from run_rbf_spectral_field import ACTIONS, FOUNDATION_TEST, OOF_BASE, OOF_META
from run_v3_anchored_neural_rbf import DESCRIPTORS


VERSION = "Round2-E23-calibrated-exact-projection"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_calibrated_exact_projection"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_calibrated_exact_projection"
METRICS = ARTIFACT_DIR / "calibrated_exact_projection_metrics.npz"
SELECTION = ARTIFACT_DIR / "calibrated_exact_projection_selection.json"
CALIBRATION_MODEL = ARTIFACT_DIR / "full_group_calibration_model.npz"
OUTPUT = ARTIFACT_DIR / "Round2_V4_Calibrated_Exact_Projection_Test_Channel.npy"
REPORT = ANALYSIS_DIR / "calibrated_exact_projection.json"

CALIBRATED_DIR = PATHS.artifacts / "round2" / "v4_predictability_calibrated_groups"
OOF_CALIBRATED_PAS = CALIBRATED_DIR / "oof_calibrated_pas_f16.npy"
OOF_CALIBRATED_PDP = CALIBRATED_DIR / "oof_calibrated_pdp_f16.npy"
OOF_V3_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "v3_anchored_neural_rbf"
    / "oof_epoch6_corrected_weights.npz"
)

TARGET_MODES = ("raw", "pascal", "pdpcal", "bothcal")
CHECKPOINTS = (1, 2, 4, 8, 16)
ORDERS = ("pas_pdp", "pdp_pas")
PAS_CALIBRATION_STRENGTH = 1.0
PAS_GROUP_BLEND = 0.25
PDP_CALIBRATION_STRENGTH = 0.5
PDP_GROUP_BLEND = 1.0
TINY = 1e-30

EXACT_PATTERN = re.compile(
    r"^exact_(raw|pascal|pdpcal|bothcal)_(pas_pdp|pdp_pas)_(.+)_c(\d+)$"
)


def candidate_names(checkpoints: tuple[int, ...]) -> list[str]:
    return [
        f"exact_{mode}_{order}_{action.name}_c{cycle:02d}"
        for mode in TARGET_MODES
        for action in ACTIONS
        for order in ORDERS
        for cycle in checkpoints
    ]


def mode_targets(
    mode: str,
    raw_pas: torch.Tensor,
    raw_pdp: torch.Tensor,
    calibrated_pas: torch.Tensor,
    calibrated_pdp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pas = calibrated_pas if mode in ("pascal", "bothcal") else raw_pas
    pdp = calibrated_pdp if mode in ("pdpcal", "bothcal") else raw_pdp
    return pas, pdp


def generate_oof_metrics(
    device: torch.device,
    batch_size: int,
    checkpoints: tuple[int, ...],
) -> dict[str, np.ndarray]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    base = np.load(OOF_BASE, mmap_mode="r")
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    ad_cache = np.load(TARGET_AD, mmap_mode="r")
    pas_cache = np.load(TARGET_PAS, mmap_mode="r")
    pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
    calibrated_pas_cache = np.load(OOF_CALIBRATED_PAS, mmap_mode="r")
    calibrated_pdp_cache = np.load(OOF_CALIBRATED_PDP, mmap_mode="r")
    names = candidate_names(checkpoints)
    output = {
        key: np.empty((len(names), len(query_ids)), dtype=np.float64)
        for key in (
            "pas",
            "pdp",
            "pas_target",
            "pdp_target",
            "error",
            "truth_energy",
            "predicted_energy",
            "cross",
        )
    }
    name_to_index = {name: index for index, name in enumerate(names)}
    max_cycles = max(checkpoints)

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        ad, raw_pas, raw_pdp = target_batch(
            ad_cache, pas_cache, pdp_cache, start, end, device
        )
        calibrated_pas = torch.as_tensor(
            np.asarray(calibrated_pas_cache[start:end], dtype=np.float32), device=device
        )
        calibrated_pdp = torch.as_tensor(
            np.asarray(calibrated_pdp_cache[start:end], dtype=np.float32), device=device
        )
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        with torch.inference_mode():
            truth_pas = (
                torch.fft.fft2(
                    truth.reshape(len(truth), 2, 16, 8, 4, 192), dim=(2, 3)
                )
                .abs()
                .square()
                .float()
            )
            truth_pdp = torch.fft.ifft(truth, dim=-1).abs().square().float()
            for mode in TARGET_MODES:
                pas, pdp = mode_targets(
                    mode, raw_pas, raw_pdp, calibrated_pas, calibrated_pdp
                )
                for action_index, action in enumerate(ACTIONS):
                    for order in ORDERS:
                        candidate, desired_pas, desired_pdp, reference_power = (
                            initialize_trajectory(
                                base_tensor, ad, pas, pdp, action_index, order
                            )
                        )
                        for cycle in range(1, max_cycles + 1):
                            if cycle > 1:
                                candidate = continue_trajectory(
                                    candidate,
                                    desired_pas,
                                    desired_pdp,
                                    reference_power,
                                    order,
                                )
                            if cycle not in checkpoints:
                                continue
                            name = (
                                f"exact_{mode}_{order}_{action.name}_c{cycle:02d}"
                            )
                            index = name_to_index[name]
                            values = metric_components(
                                candidate,
                                truth,
                                truth_pas,
                                truth_pdp,
                                desired_pas,
                                desired_pdp,
                            )
                            for key, value in values.items():
                                output[key][index, start:end] = value
        print(f"calibrated exact OOF {end}/{len(query_ids)}", flush=True)

    output["candidate_names"] = np.asarray(names)
    output["query_ids"] = query_ids
    output["fold_id"] = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    output["site"] = train_site[query_ids].astype(np.int8)
    return output


def evaluate_final_policy(
    combined: dict[str, np.ndarray], final_policy: list[dict[str, object]]
) -> dict[str, object]:
    names = combined["candidate_names"]
    site = combined["site"]
    fold_id = combined["fold_id"]
    selected = np.empty(len(site), dtype=np.int64)
    scale = np.empty(len(site), dtype=np.float64)
    for row in final_policy:
        mask = site == int(row["site"])
        candidate_index = int(np.flatnonzero(names == row["candidate"])[0])
        selected[mask] = candidate_index
        scale[mask] = float(row["scale"])
    point = np.arange(len(site))
    error = (
        scale * scale * combined["predicted_energy"][selected, point]
        + combined["truth_energy"][selected, point]
        - 2.0 * scale * combined["cross"][selected, point]
    )
    error = np.maximum(error, 0.0)
    pas = combined["pas"][selected, point]
    pdp = combined["pdp"][selected, point]
    energy = combined["truth_energy"][selected, point]
    from run_rbf_spectral_field import statistics

    return {
        "overall": statistics(pas, pdp, error, energy),
        "folds": [
            {
                "fold": int(fold),
                **statistics(pas, pdp, error, energy, fold_id == fold),
            }
            for fold in np.unique(fold_id)
        ],
    }


def calibration_slope(
    amplitude: np.ndarray,
    prediction: np.ndarray,
    query_ids: np.ndarray,
    site: np.ndarray,
    query_site: np.ndarray,
    valid: np.ndarray,
    components: int,
    device: torch.device,
    seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    flat = np.asarray(amplitude, dtype=np.float32).reshape(len(amplitude), -1)
    models: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for site_value in np.unique(site):
        fit_ids = np.flatnonzero(valid & (site == site_value))
        rows = np.flatnonzero(query_site == site_value)
        mean, vectors = pca_basis(
            flat[fit_ids], components, device, seed + int(site_value) * 1009
        )
        predicted_coefficients = encode(
            prediction[rows].reshape(len(rows), -1), mean, vectors
        )
        truth_coefficients = encode(flat[query_ids[rows]], mean, vectors)
        denominator = np.sum(
            predicted_coefficients * predicted_coefficients,
            axis=0,
            dtype=np.float64,
        )
        slope = np.divide(
            np.sum(
                predicted_coefficients * truth_coefficients,
                axis=0,
                dtype=np.float64,
            ),
            np.maximum(denominator, 1e-12),
        )
        models[int(site_value)] = (
            mean.astype(np.float32),
            vectors.astype(np.float32),
            np.clip(slope, 0.0, 1.25).astype(np.float32),
        )
    return models


def apply_calibration(
    prediction: np.ndarray,
    query_site: np.ndarray,
    models: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    strength: float,
) -> np.ndarray:
    output = np.empty_like(prediction, dtype=np.float32)
    shape = prediction.shape[1:]
    for site_value, (mean, vectors, slope) in models.items():
        rows = np.flatnonzero(query_site == site_value)
        if len(rows) == 0:
            continue
        flat = prediction[rows].reshape(len(rows), -1)
        coefficients = encode(flat, mean, vectors)
        gain = 1.0 + strength * (slope - 1.0)
        corrected = flat + ((coefficients * gain) - coefficients) @ vectors.T
        output[rows] = normalize_group_amplitude(corrected.reshape(len(rows), *shape))
    return output


def build_full_calibration(
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    _, _, test_positions, train_site, test_site = load_positions()
    del test_positions
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    weights = np.load(OOF_V3_WEIGHTS)
    test_weights = np.load(TEST_WEIGHTS)
    pas_amplitude = np.load(GROUPED_PAS, mmap_mode="r")
    pdp_amplitude = np.load(GROUPED_PDP, mmap_mode="r")

    oof_pas = predict_grouped(
        pas_amplitude,
        weights["donor_ids"].astype(np.int64),
        weights["pas_weights"].astype(np.float32),
    )
    oof_pdp = predict_grouped(
        pdp_amplitude,
        weights["donor_ids"].astype(np.int64),
        weights["pdp_weights"].astype(np.float32),
    )
    query_site = train_site[query_ids]
    pas_models = calibration_slope(
        pas_amplitude,
        oof_pas,
        query_ids,
        train_site,
        query_site,
        valid,
        PAS_COMPONENTS,
        device,
        20_260_805,
    )
    pdp_models = calibration_slope(
        pdp_amplitude,
        oof_pdp,
        query_ids,
        train_site,
        query_site,
        valid,
        PDP_COMPONENTS,
        device,
        20_270_805,
    )

    test_pas = predict_grouped(
        pas_amplitude,
        test_weights["donor_ids"].astype(np.int64),
        test_weights["pas_weights"].astype(np.float32),
    )
    test_pdp = predict_grouped(
        pdp_amplitude,
        test_weights["donor_ids"].astype(np.int64),
        test_weights["pdp_weights"].astype(np.float32),
    )
    calibrated_pas = apply_calibration(
        test_pas, test_site, pas_models, PAS_CALIBRATION_STRENGTH
    )
    calibrated_pdp = apply_calibration(
        test_pdp, test_site, pdp_models, PDP_CALIBRATION_STRENGTH
    )

    arrays: dict[str, np.ndarray] = {
        "test_grouped_pas": calibrated_pas,
        "test_grouped_pdp": calibrated_pdp,
    }
    diagnostics: dict[str, object] = {"sites": {}}
    model_arrays: dict[str, np.ndarray] = {}
    for site_value in sorted(pas_models):
        pas_mean, pas_vectors, pas_slope = pas_models[site_value]
        pdp_mean, pdp_vectors, pdp_slope = pdp_models[site_value]
        prefix = f"site{site_value}"
        model_arrays[f"{prefix}_pas_mean"] = pas_mean
        model_arrays[f"{prefix}_pas_vectors"] = pas_vectors
        model_arrays[f"{prefix}_pas_slope"] = pas_slope
        model_arrays[f"{prefix}_pdp_mean"] = pdp_mean
        model_arrays[f"{prefix}_pdp_vectors"] = pdp_vectors
        model_arrays[f"{prefix}_pdp_slope"] = pdp_slope
        diagnostics["sites"][str(site_value)] = {
            "pas_slope_mean": float(np.mean(pas_slope)),
            "pdp_slope_mean": float(np.mean(pdp_slope)),
            "pas_components": int(pas_vectors.shape[1]),
            "pdp_components": int(pdp_vectors.shape[1]),
        }
    np.savez_compressed(CALIBRATION_MODEL, **model_arrays)
    return arrays, diagnostics


def action_by_name(name: str):
    for action in ACTIONS:
        if action.name == name:
            return action
    raise KeyError(name)


def generate_test_output(
    selection: dict[str, object],
    calibration: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    weights = np.load(TEST_WEIGHTS)
    donor_ids = weights["donor_ids"].astype(np.int64)
    pas_weights = weights["pas_weights"].astype(np.float32)
    pdp_weights = weights["pdp_weights"].astype(np.float32)
    train_ad = np.load(TRAIN_AD, mmap_mode="r")
    train_pas = np.load(TRAIN_PAS, mmap_mode="r")
    train_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    base = np.load(FOUNDATION_TEST, mmap_mode="r")
    _, _, test_positions, _, test_site = load_positions()
    policy = {int(row["site"]): row for row in selection["final_policy"]}
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )

    for start in range(0, len(test_positions), batch_size):
        end = min(start + batch_size, len(test_positions))
        ad_rows: list[np.ndarray] = []
        pas_rows: list[np.ndarray] = []
        pdp_rows: list[np.ndarray] = []
        for row in range(start, end):
            ad_rows.append(
                spectral_target(
                    train_ad, donor_ids[row], pas_weights[row], (0, 1, 2)
                )
            )
            pas_rows.append(
                spectral_target(train_pas, donor_ids[row], pas_weights[row], 0)
            )
            pdp_rows.append(
                spectral_target(train_pdp, donor_ids[row], pdp_weights[row], 2)
            )
        raw_pas, raw_pdp = globally_shrunk_targets(
            np.stack(pas_rows), np.stack(pdp_rows)
        )
        calibrated_pas = blend_group_pas(
            raw_pas,
            calibration["test_grouped_pas"][start:end],
            PAS_GROUP_BLEND,
        )
        calibrated_pdp = blend_group_pdp(
            raw_pdp,
            calibration["test_grouped_pdp"][start:end],
            PDP_GROUP_BLEND,
        )
        ad_tensor = torch.as_tensor(np.stack(ad_rows), device=device)
        raw_pas_tensor = torch.as_tensor(raw_pas, device=device)
        raw_pdp_tensor = torch.as_tensor(raw_pdp, device=device)
        calibrated_pas_tensor = torch.as_tensor(calibrated_pas, device=device)
        calibrated_pdp_tensor = torch.as_tensor(calibrated_pdp, device=device)
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        block = np.empty_like(np.asarray(base[start:end]))
        block_site = test_site[start:end]
        with torch.inference_mode():
            for site_value in np.unique(block_site):
                local = np.flatnonzero(block_site == site_value)
                row = policy[int(site_value)]
                name = str(row["candidate"])
                if name.startswith("legacy_"):
                    action = action_by_name(name.removeprefix("legacy_"))
                    prediction = project_channel(
                        base_tensor[local],
                        ad_tensor[local],
                        raw_pas_tensor[local],
                        raw_pdp_tensor[local],
                        action.alpha_ad,
                        action.alpha_pas,
                        action.alpha_pdp,
                    )
                else:
                    match = EXACT_PATTERN.fullmatch(name)
                    if match is None:
                        raise ValueError(f"cannot parse candidate {name}")
                    mode, order, action_name, cycles_text = match.groups()
                    action_index = next(
                        index
                        for index, action in enumerate(ACTIONS)
                        if action.name == action_name
                    )
                    pas, pdp = mode_targets(
                        mode,
                        raw_pas_tensor[local],
                        raw_pdp_tensor[local],
                        calibrated_pas_tensor[local],
                        calibrated_pdp_tensor[local],
                    )
                    prediction, desired_pas, desired_pdp, reference_power = (
                        initialize_trajectory(
                            base_tensor[local],
                            ad_tensor[local],
                            pas,
                            pdp,
                            action_index,
                            order,
                        )
                    )
                    for _ in range(2, int(cycles_text) + 1):
                        prediction = continue_trajectory(
                            prediction,
                            desired_pas,
                            desired_pdp,
                            reference_power,
                            order,
                        )
                block[local] = (
                    prediction.cpu().numpy() * np.float32(row["scale"])
                ).astype(np.complex64)
        output[start:end] = block
        output.flush()
        print(f"V4 test {end}/{len(test_positions)}", flush=True)

    finite = True
    for start in range(0, len(output), 8):
        value = np.asarray(output[start : start + 8])
        finite = finite and bool(
            np.isfinite(value.real).all() and np.isfinite(value.imag).all()
        )
    if not finite:
        raise ValueError("V4 output contains non-finite values")
    return {
        "path": PATHS.relative(OUTPUT),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(OUTPUT.stat().st_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-cycles", type=int, choices=CHECKPOINTS, default=16)
    parser.add_argument("--reuse-metrics", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = tuple(value for value in CHECKPOINTS if value <= args.max_cycles)
    device = torch.device(args.device)
    if args.reuse_metrics:
        generated = {key: value for key, value in np.load(METRICS).items()}
    else:
        generated = generate_oof_metrics(device, args.batch_size, checkpoints)
        np.savez_compressed(METRICS, **generated)
    combined = combine_with_v3(generated)
    selected = select_nested(combined)
    baseline = v3_nested_baseline(combined)
    final_policy_score = evaluate_final_policy(combined, selected["final_policy"])
    candidate_rows = summarize_candidates(generated, combined)
    fold_deltas = [
        {
            "fold": int(new["fold"]),
            "score": float(new["score"] - old["score"]),
            "pas": float(new["pas"] - old["pas"]),
            "pdp": float(new["pdp"] - old["pdp"]),
            "nmse": float(new["nmse"] - old["nmse"]),
        }
        for new, old in zip(selected["folds"], baseline["folds"])
    ]
    selection_payload = {
        "experiment": VERSION,
        "final_policy": selected["final_policy"],
        "nested_selected": selected["overall"],
        "nested_folds": selected["folds"],
        "nested_decisions": selected["decisions"],
        "final_policy_oof": final_policy_score,
    }
    SELECTION.write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    calibration_diagnostics: dict[str, object] | None = None
    output: dict[str, object] | None = None
    if not args.skip_test:
        calibration, calibration_diagnostics = build_full_calibration(device)
        output = generate_test_output(
            selection_payload, calibration, device, args.batch_size
        )
    passed = (
        selected["overall"]["score"] > baseline["overall"]["score"]
        and sum(row["score"] > 0.0 for row in fold_deltas) >= 4
    )
    report = {
        "experiment": VERSION,
        "status": "candidate_ready" if passed and output is not None else "analysis_only",
        "primary_hypothesis": (
            "V3's grouped PDP contains spatially predictable low-rank modes, while exact "
            "alternating PAS/PDP magnitude projections recover target structure lost by the "
            "single clipped projection."
        ),
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": (
            "fixed 500-point five-fold spatial holes; every OOF target, PCA basis, "
            "reliability slope, candidate decision and scale excludes the held spatial hole"
        ),
        "parameters": {
            "target_modes": list(TARGET_MODES),
            "actions": [asdict(action) for action in ACTIONS],
            "orders": list(ORDERS),
            "checkpoints": list(checkpoints),
            "pas_calibration_strength": PAS_CALIBRATION_STRENGTH,
            "pas_group_blend": PAS_GROUP_BLEND,
            "pdp_calibration_strength": PDP_CALIBRATION_STRENGTH,
            "pdp_group_blend": PDP_GROUP_BLEND,
        },
        "v3_nested_baseline": baseline,
        "nested_selected": {
            "overall": selected["overall"],
            "folds": selected["folds"],
            "decisions": selected["decisions"],
            "final_policy": selected["final_policy"],
        },
        "final_policy_oof": final_policy_score,
        "delta_vs_v3": {
            "overall_score": float(
                selected["overall"]["score"] - baseline["overall"]["score"]
            ),
            "folds": fold_deltas,
        },
        "top_candidates": candidate_rows[:20],
        "calibration_diagnostics": calibration_diagnostics,
        "output": output,
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "selection": PATHS.relative(SELECTION),
            "calibration_model": PATHS.relative(CALIBRATION_MODEL),
            "test_channel": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "v3": baseline["overall"],
                "selected": selected["overall"],
                "fold_deltas": fold_deltas,
                "final_policy": selected["final_policy"],
                "final_policy_oof": final_policy_score["overall"],
                "output": output,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
