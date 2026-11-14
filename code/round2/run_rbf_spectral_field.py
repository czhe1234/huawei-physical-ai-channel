#!/usr/bin/env python3
"""Build and validate a square-root RBF spectral field for Round 2."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree, distance_matrix

from data import load_positions
from official_metrics import PointMetricStatistics, aggregate_statistics, point_metric_statistics
from pre_sionna_spectral import project_channel
from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "rbf_spectral_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_rbf_spectral_field"
OUTPUT = ARTIFACT_DIR / "Round2_RBF_Spectral_Field_Test_Channel.npy"
WEIGHTS = ARTIFACT_DIR / "round2_rbf_test_weights.npz"
OOF_METRICS = ARTIFACT_DIR / "round2_rbf_oof_metrics.npz"
SELECTION = ARTIFACT_DIR / "round2_rbf_selection.json"
REPORT = ANALYSIS_DIR / "round2_rbf_spectral_field.json"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
TRAIN_AD = PRE_DIR / "round2_train_ad_support_f16.npy"
TRAIN_PAS = PRE_DIR / "round2_train_pas_link_support_f16.npy"
TRAIN_PDP = PRE_DIR / "round2_train_pdp_link_support_f16.npy"
TRAIN_DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
OOF_BASE = PRE_DIR / "round2_pre_sionna_oof_foundation.npy"
OOF_META = PRE_DIR / "round2_pre_sionna_oof_foundation_meta.npz"
OOF_WEIGHTS = (
    ARTIFACT_DIR / "round2_rbf_oof_weights.npz"
)
FOUNDATION_TEST = (
    PATHS.artifacts / "round2" / "foundation" / "Round2_Foundation_Test_Channel.npy"
)
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)


@dataclass(frozen=True)
class ProjectionAction:
    name: str
    alpha_ad: float
    alpha_pas: float
    alpha_pdp: float


ACTIONS = (
    ProjectionAction("ad000_pas090_pdp090", 0.00, 0.90, 0.90),
    ProjectionAction("ad050_pas075_pdp080", 0.50, 0.75, 0.80),
    ProjectionAction("ad050_pas090_pdp090", 0.50, 0.90, 0.90),
    ProjectionAction("ad075_pas090_pdp090", 0.75, 0.90, 0.90),
    ProjectionAction("ad100_pas090_pdp090", 1.00, 0.90, 0.90),
    ProjectionAction("ad075_pas100_pdp100", 0.75, 1.00, 1.00),
    ProjectionAction("ad100_pas100_pdp100", 1.00, 1.00, 1.00),
)


def normalized(value: np.ndarray, axes: int | tuple[int, ...]) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, 1e-20),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def solve_linear_rbf_weights(
    donor_xy: np.ndarray,
    query_xy: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    count = len(donor_xy)
    system = np.empty((count + 1, count + 1), dtype=np.float64)
    system[:count, :count] = -distance_matrix(donor_xy, donor_xy)
    system[np.arange(count), np.arange(count)] += smoothing
    system[:count, count] = 1.0
    system[count, :count] = 1.0
    system[count, count] = 0.0
    right = np.concatenate(
        [-np.linalg.norm(donor_xy - query_xy[None, :], axis=1), np.ones(1)]
    )
    weights = np.linalg.solve(system, right)[:count]
    return weights.astype(np.float32)


def build_weights(
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    train_site: np.ndarray,
    query_site: np.ndarray,
    valid: np.ndarray,
    neighbors: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    donor_ids = np.empty((len(query_positions), neighbors), dtype=np.int32)
    weights = np.empty((len(query_positions), neighbors), dtype=np.float32)
    for site_value in np.unique(query_site):
        rows = np.flatnonzero(query_site == site_value)
        bank = np.flatnonzero((train_site == site_value) & valid)
        _, local = cKDTree(train_positions[bank, :2]).query(
            query_positions[rows, :2], k=neighbors, workers=-1
        )
        selected = bank[local]
        donor_ids[rows] = selected
        for local_row, output_row in enumerate(rows):
            weights[output_row] = solve_linear_rbf_weights(
                train_positions[selected[local_row], :2],
                query_positions[output_row, :2],
                smoothing,
            )
    return donor_ids, weights


def build_oof_weights(
    train_positions: np.ndarray,
    train_site: np.ndarray,
    valid: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    neighbors: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build RBF donors from each held fold's fit-only bank."""

    folds = np.load(FOLD_CACHE)
    donor_ids = np.empty((len(query_ids), neighbors), dtype=np.int32)
    weights = np.empty((len(query_ids), neighbors), dtype=np.float32)
    for held_fold in np.unique(fold_id):
        rows_in_fold = np.flatnonzero(fold_id == held_fold)
        fit_ids = folds[f"fold{int(held_fold)}_fit"].astype(np.int64)
        for site_value in np.unique(train_site[query_ids[rows_in_fold]]):
            rows = rows_in_fold[
                train_site[query_ids[rows_in_fold]] == site_value
            ]
            bank = fit_ids[(train_site[fit_ids] == site_value) & valid[fit_ids]]
            if len(bank) < neighbors:
                raise RuntimeError(
                    f"site {site_value} fold {held_fold} has only {len(bank)} donors"
                )
            _, local = cKDTree(train_positions[bank, :2]).query(
                train_positions[query_ids[rows], :2],
                k=neighbors,
                workers=-1,
            )
            selected = bank[local]
            donor_ids[rows] = selected
            for local_row, output_row in enumerate(rows):
                weights[output_row] = solve_linear_rbf_weights(
                    train_positions[selected[local_row], :2],
                    train_positions[query_ids[output_row], :2],
                    smoothing,
                )
    np.savez_compressed(
        OOF_WEIGHTS,
        neighbor_ids=donor_ids,
        weights=weights,
        query_ids=query_ids,
        fold_id=fold_id,
        neighbors=np.int32(neighbors),
        smoothing=np.float32(smoothing),
    )
    return donor_ids, weights


def load_or_build_oof_weights(
    train_positions: np.ndarray,
    train_site: np.ndarray,
    valid: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    neighbors: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    if OOF_WEIGHTS.is_file():
        cached = np.load(OOF_WEIGHTS)
        compatible = (
            np.array_equal(cached["query_ids"], query_ids)
            and np.array_equal(cached["fold_id"], fold_id)
            and int(cached["neighbors"]) == neighbors
            and np.isclose(float(cached["smoothing"]), smoothing)
        )
        if compatible:
            return (
                cached["neighbor_ids"].astype(np.int64),
                cached["weights"].astype(np.float32),
            )
    return build_oof_weights(
        train_positions,
        train_site,
        valid,
        query_ids,
        fold_id,
        neighbors,
        smoothing,
    )


def spectral_target(
    cache: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    axes: int | tuple[int, ...],
) -> np.ndarray:
    donor = np.asarray(cache[donor_ids], dtype=np.float32)
    square_root = np.sqrt(np.maximum(donor, 0.0))
    interpolated = np.square(
        np.maximum(
            np.tensordot(weights.astype(np.float32), square_root, axes=(0, 0)),
            0.0,
        )
    )
    return normalized(interpolated, axes)


def target_batch(
    train_ad: np.ndarray,
    train_pas: np.ndarray,
    train_pdp: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ad: list[np.ndarray] = []
    pas: list[np.ndarray] = []
    pdp: list[np.ndarray] = []
    for ids, row_weights in zip(donor_ids, weights):
        ad.append(spectral_target(train_ad, ids, row_weights, (0, 1, 2)))
        pas.append(spectral_target(train_pas, ids, row_weights, 0))
        pdp.append(spectral_target(train_pdp, ids, row_weights, 2))
    return np.stack(ad), np.stack(pas), np.stack(pdp)


def optimal_scale(
    predicted_energy: np.ndarray,
    cross: np.ndarray,
    mask: np.ndarray,
) -> float:
    denominator = float(np.sum(predicted_energy[mask], dtype=np.float64))
    numerator = float(np.sum(cross[mask], dtype=np.float64))
    if denominator <= 1e-30:
        return 1.0
    return float(np.clip(numerator / denominator, 0.05, 1.50))


def scaled_error(
    predicted_energy: np.ndarray,
    truth_energy: np.ndarray,
    cross: np.ndarray,
    scale: np.ndarray | float,
) -> np.ndarray:
    value = (
        np.asarray(scale) ** 2 * predicted_energy
        + truth_energy
        - 2.0 * np.asarray(scale) * cross
    )
    return np.maximum(value, 0.0)


def statistics(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    truth_energy: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    return aggregate_statistics(
        PointMetricStatistics(pas, pdp, error, truth_energy), mask
    )


def evaluate_oof(
    device: torch.device,
    batch_size: int,
    neighbors: int,
    smoothing: float,
) -> dict[str, object]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, train_positions, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]
    descriptors = np.load(TRAIN_DESCRIPTORS)
    donor_ids, weights = load_or_build_oof_weights(
        train_positions,
        train_site,
        descriptors["valid"].astype(bool),
        query_ids,
        fold_id,
        neighbors,
        smoothing,
    )
    train_ad = np.load(TRAIN_AD, mmap_mode="r")
    train_pas = np.load(TRAIN_PAS, mmap_mode="r")
    train_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    base = np.load(OOF_BASE, mmap_mode="r")
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    count = len(query_ids)
    shape = (len(ACTIONS), count)
    pas_values = np.empty(shape, dtype=np.float64)
    pdp_values = np.empty(shape, dtype=np.float64)
    error_values = np.empty(shape, dtype=np.float64)
    truth_energy_values = np.empty(shape, dtype=np.float64)
    predicted_energy = np.empty(shape, dtype=np.float64)
    cross = np.empty(shape, dtype=np.float64)

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        ad, pas, pdp = target_batch(
            train_ad,
            train_pas,
            train_pdp,
            donor_ids[start:end],
            weights[start:end],
        )
        base_numpy = np.array(base[start:end], dtype=np.complex64, copy=True)
        truth = np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True)
        base_tensor = torch.as_tensor(base_numpy, device=device)
        ad_tensor = torch.as_tensor(ad, device=device)
        pas_tensor = torch.as_tensor(pas, device=device)
        pdp_tensor = torch.as_tensor(pdp, device=device)
        for action_index, action in enumerate(ACTIONS):
            with torch.inference_mode():
                prediction = project_channel(
                    base_tensor,
                    ad_tensor,
                    pas_tensor,
                    pdp_tensor,
                    action.alpha_ad,
                    action.alpha_pas,
                    action.alpha_pdp,
                ).cpu().numpy()
            metric = point_metric_statistics(prediction, truth)
            pas_values[action_index, start:end] = metric.pas
            pdp_values[action_index, start:end] = metric.pdp
            error_values[action_index, start:end] = metric.squared_error
            truth_energy_values[action_index, start:end] = metric.truth_energy
            predicted_energy[action_index, start:end] = np.sum(
                np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64
            )
            cross[action_index, start:end] = np.real(
                np.sum(
                    prediction * np.conj(truth),
                    axis=(1, 2, 3),
                    dtype=np.complex128,
                )
            )
        print(f"RBF OOF {end}/{count}", flush=True)

    raw_actions: list[dict[str, object]] = []
    calibrated_actions: list[dict[str, object]] = []
    for action_index, action in enumerate(ACTIONS):
        raw_actions.append(
            {
                "action": asdict(action),
                "all": statistics(
                    pas_values[action_index],
                    pdp_values[action_index],
                    error_values[action_index],
                    truth_energy_values[action_index],
                ),
            }
        )
        scale_by_site: dict[str, float] = {}
        calibrated = error_values[action_index].copy()
        for site_value in np.unique(query_site):
            mask = query_site == site_value
            scale = optimal_scale(
                predicted_energy[action_index], cross[action_index], mask
            )
            scale_by_site[str(int(site_value))] = scale
            calibrated[mask] = scaled_error(
                predicted_energy[action_index, mask],
                truth_energy_values[action_index, mask],
                cross[action_index, mask],
                scale,
            )
        calibrated_actions.append(
            {
                "action": asdict(action),
                "scale_by_site": scale_by_site,
                "all": statistics(
                    pas_values[action_index],
                    pdp_values[action_index],
                    calibrated,
                    truth_energy_values[action_index],
                ),
            }
        )

    selected_action = np.empty(count, dtype=np.int16)
    selected_scale = np.empty(count, dtype=np.float32)
    nested_error = np.empty(count, dtype=np.float64)
    decisions: list[dict[str, object]] = []
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(query_site):
            train = (fold_id != held_fold) & (query_site == site_value)
            held = (fold_id == held_fold) & (query_site == site_value)
            candidates: list[tuple[float, int, float]] = []
            for action_index in range(len(ACTIONS)):
                scale = optimal_scale(
                    predicted_energy[action_index], cross[action_index], train
                )
                error = scaled_error(
                    predicted_energy[action_index],
                    truth_energy_values[action_index],
                    cross[action_index],
                    scale,
                )
                score = float(
                    statistics(
                        pas_values[action_index],
                        pdp_values[action_index],
                        error,
                        truth_energy_values[action_index],
                        train,
                    )["score"]
                )
                candidates.append((score, action_index, scale))
            train_score, action_index, scale = max(candidates)
            selected_action[held] = action_index
            selected_scale[held] = scale
            nested_error[held] = scaled_error(
                predicted_energy[action_index, held],
                truth_energy_values[action_index, held],
                cross[action_index, held],
                scale,
            )
            decisions.append(
                {
                    "held_fold": int(held_fold),
                    "site": int(site_value),
                    "action": ACTIONS[action_index].name,
                    "scale": float(scale),
                    "training_score": train_score,
                    "held_count": int(np.sum(held)),
                }
            )

    selected_pas = np.empty(count, dtype=np.float64)
    selected_pdp = np.empty(count, dtype=np.float64)
    selected_truth_energy = np.empty(count, dtype=np.float64)
    for action_index in range(len(ACTIONS)):
        mask = selected_action == action_index
        selected_pas[mask] = pas_values[action_index, mask]
        selected_pdp[mask] = pdp_values[action_index, mask]
        selected_truth_energy[mask] = truth_energy_values[action_index, mask]
    nested = statistics(
        selected_pas, selected_pdp, nested_error, selected_truth_energy
    )

    final_policy: list[dict[str, object]] = []
    for site_value in np.unique(query_site):
        mask = query_site == site_value
        candidates = []
        for action_index in range(len(ACTIONS)):
            scale = optimal_scale(
                predicted_energy[action_index], cross[action_index], mask
            )
            error = scaled_error(
                predicted_energy[action_index],
                truth_energy_values[action_index],
                cross[action_index],
                scale,
            )
            score = float(
                statistics(
                    pas_values[action_index],
                    pdp_values[action_index],
                    error,
                    truth_energy_values[action_index],
                    mask,
                )["score"]
            )
            candidates.append((score, action_index, scale))
        score, action_index, scale = max(candidates)
        final_policy.append(
            {
                "site": int(site_value),
                "action_index": int(action_index),
                "action": ACTIONS[action_index].name,
                "scale": float(scale),
                "oof_site_score": score,
            }
        )

    np.savez_compressed(
        OOF_METRICS,
        action_names=np.asarray([action.name for action in ACTIONS]),
        pas=pas_values,
        pdp=pdp_values,
        error=error_values,
        truth_energy=truth_energy_values,
        predicted_energy=predicted_energy,
        cross=cross,
        query_ids=query_ids,
        fold_id=fold_id,
        site=query_site,
        selected_action=selected_action,
        selected_scale=selected_scale,
    )
    payload = {
        "raw_actions": raw_actions,
        "calibrated_actions": calibrated_actions,
        "nested_selected": nested,
        "nested_decisions": decisions,
        "final_policy": final_policy,
    }
    SELECTION.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def generate_test(
    device: torch.device,
    batch_size: int,
    selection: dict[str, object],
    neighbors: int,
    smoothing: float,
) -> dict[str, object]:
    _, train_positions, test_positions, train_site, test_site = load_positions()
    descriptors = np.load(TRAIN_DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    donor_ids, weights = build_weights(
        train_positions,
        test_positions,
        train_site,
        test_site,
        valid,
        neighbors,
        smoothing,
    )
    np.savez_compressed(
        WEIGHTS,
        donor_ids=donor_ids,
        weights=weights,
        neighbors=np.int32(neighbors),
        smoothing=np.float32(smoothing),
    )
    train_ad = np.load(TRAIN_AD, mmap_mode="r")
    train_pas = np.load(TRAIN_PAS, mmap_mode="r")
    train_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    base = np.load(FOUNDATION_TEST, mmap_mode="r")
    policy = {int(row["site"]): row for row in selection["final_policy"]}
    output = np.lib.format.open_memmap(
        OUTPUT,
        mode="w+",
        dtype=np.complex64,
        shape=base.shape,
    )
    for start in range(0, len(test_positions), batch_size):
        end = min(start + batch_size, len(test_positions))
        ad, pas, pdp = target_batch(
            train_ad,
            train_pas,
            train_pdp,
            donor_ids[start:end],
            weights[start:end],
        )
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        ad_tensor = torch.as_tensor(ad, device=device)
        pas_tensor = torch.as_tensor(pas, device=device)
        pdp_tensor = torch.as_tensor(pdp, device=device)
        block = np.empty_like(np.asarray(base[start:end]))
        block_site = test_site[start:end]
        for site_value in np.unique(block_site):
            local = np.flatnonzero(block_site == site_value)
            row = policy[int(site_value)]
            action = ACTIONS[int(row["action_index"])]
            with torch.inference_mode():
                prediction = project_channel(
                    base_tensor[local],
                    ad_tensor[local],
                    pas_tensor[local],
                    pdp_tensor[local],
                    action.alpha_ad,
                    action.alpha_pas,
                    action.alpha_pdp,
                )
            block[local] = (
                prediction.cpu().numpy() * np.float32(row["scale"])
            ).astype(np.complex64)
        output[start:end] = block
        output.flush()
        print(f"RBF test {end}/{len(test_positions)}", flush=True)
    finite = True
    for start in range(0, len(output), 8):
        block = np.asarray(output[start : start + 8])
        finite = finite and bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
    if not finite:
        raise ValueError("RBF test output contains non-finite values")
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--smoothing", type=float, default=10.0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.skip_validation:
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    else:
        selection = evaluate_oof(
            device,
            args.batch_size,
            args.neighbors,
            args.smoothing,
        )
    output = None
    if not args.skip_generate:
        output = generate_test(
            device,
            args.batch_size,
            selection,
            args.neighbors,
            args.smoothing,
        )
    report = {
        "experiment": "Round2-E3",
        "status": "validated_candidate_source",
        "primary_hypothesis": (
            "Linear RBF interpolation in square-root spectral space preserves moving path "
            "peaks better than direct power-template averaging across spatial holes."
        ),
        "sionna_used": False,
        "official_inputs_only": True,
        "neighbors": args.neighbors,
        "smoothing": args.smoothing,
        "actions": [asdict(action) for action in ACTIONS],
        "validation": selection,
        "output": output,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
