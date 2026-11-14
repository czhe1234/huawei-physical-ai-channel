#!/usr/bin/env python3
"""Test path-specific carrier-phase transport on matched official-H paths."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v309_unmatched_effective_scatterers as v309
import analyze_v315_full_sequence_confirmatory as v315
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 325
VERSION = "v325_path_phase_transport_identifiability"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CENTER_GAIN_CACHE = OUT_DIR / "v325_component_center_gains.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

CANDIDATE_CACHE = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
BETA_GRID = np.arange(120.0, 160.0001, 0.02, dtype=np.float64)
MAX_PAIR_DISTANCE = 25.0
MIN_CENTER_COHERENCE = 0.65
SHUFFLE_REPEATS = 20


def recover_center_gains() -> dict[str, np.ndarray]:
    if CENTER_GAIN_CACHE.is_file():
        data = np.load(CENTER_GAIN_CACHE)
        return {key: data[key] for key in data.files}

    component = np.load(v297.COMPONENT_CACHE)
    delays = component["delay"].astype(np.float64)
    valid = component["valid"].astype(bool)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    center_gain = np.zeros(delays.shape + (2, 4), dtype=np.complex64)
    center_coherence = np.zeros(delays.shape + (2, 4), dtype=np.float32)
    frequency = np.arange(v297.N_DELAY, dtype=np.float64)
    h_coordinate = np.arange(16, dtype=np.float32) - 7.5
    v_coordinate = np.arange(8, dtype=np.float32) - 3.5

    for point_id in range(len(channel)):
        active = np.flatnonzero(valid[point_id])
        if len(active) == 0:
            continue
        tau = delays[point_id, active]
        steering = np.exp(
            -2j * np.pi * frequency[:, None] * tau[None, :] / v297.N_DELAY
        ).astype(np.complex64)
        h = np.asarray(channel[point_id], dtype=np.complex64).reshape(-1, v297.N_DELAY)
        gram = steering.conj().T @ steering
        regularized = gram + np.eye(len(active), dtype=np.complex64) * (
            v300.RIDGE * np.trace(gram).real / max(len(active), 1)
        )
        gain = np.linalg.solve(regularized, steering.conj().T @ h.T)
        tensor = gain.reshape(len(active), 2, 16, 8, 4)
        for polarization in range(2):
            for ue_port in range(4):
                array = tensor[:, polarization, :, :, ue_port]
                h_step = np.angle(
                    np.sum(array[:, 1:, :] * np.conj(array[:, :-1, :]), axis=(1, 2))
                )
                v_step = np.angle(
                    np.sum(array[:, :, 1:] * np.conj(array[:, :, :-1]), axis=(1, 2))
                )
                phase = (
                    h_step[:, None, None] * h_coordinate[None, :, None]
                    + v_step[:, None, None] * v_coordinate[None, None, :]
                )
                demodulated = array * np.exp(-1j * phase)
                center = np.mean(demodulated, axis=(1, 2))
                mean_power = np.mean(np.abs(array) ** 2, axis=(1, 2))
                coherence = np.abs(center) ** 2 / np.maximum(mean_power, 1e-30)
                center_gain[point_id, active, polarization, ue_port] = center
                center_coherence[point_id, active, polarization, ue_port] = coherence
        if (point_id + 1) % 200 == 0 or point_id + 1 == len(channel):
            print(f"v325 center gains {point_id + 1}/{len(channel)}", flush=True)

    np.savez_compressed(
        CENTER_GAIN_CACHE,
        center_gain=center_gain,
        center_coherence=center_coherence,
        valid=valid,
    )
    return {
        "center_gain": center_gain,
        "center_coherence": center_coherence,
        "valid": valid,
    }


def matched_rows(gain_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    match = np.load(v305.MATCH_CACHE)
    candidates = np.load(CANDIDATE_CACHE)
    good = (
        (match["length_error_bins"] <= v309.GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= v309.GOOD_H_ERROR)
    )
    point = match["point_id"][good].astype(np.int64)
    component = match["component_id"][good].astype(np.int64)
    candidate = match["candidate_id"][good].astype(np.int64)
    patch = match["patch_id"][good].astype(np.int64)
    mass = match["mass"][good].astype(np.float32)
    length = candidates["path_length"][point, candidate].astype(np.float32)
    h_center = v305.expected_h_bin(
        candidates["reflection_point"][point, candidate]
    ).astype(np.float32)
    center_gain = gain_data["center_gain"][point, component]
    coherence = gain_data["center_coherence"][point, component]

    # A point can occasionally assign more than one component to the same wall.
    # Keep the strongest one so phase pairs represent one propagation family.
    selected: dict[tuple[int, int], int] = {}
    for row, key in enumerate(zip(point.tolist(), patch.tolist())):
        previous = selected.get(key)
        if previous is None or mass[row] > mass[previous]:
            selected[key] = row
    keep = np.asarray(sorted(selected.values()), dtype=np.int64)
    return {
        "point": point[keep],
        "component": component[keep],
        "candidate": candidate[keep],
        "patch": patch[keep],
        "mass": mass[keep],
        "length": length[keep],
        "h_center": h_center[keep],
        "gain": center_gain[keep],
        "coherence": coherence[keep],
    }


def nearest_patch_pairs(
    rows: dict[str, np.ndarray],
    query_points: np.ndarray,
    bank_points: np.ndarray,
    xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_mask = np.isin(rows["point"], query_points)
    bank_mask = np.isin(rows["point"], bank_points)
    query_rows = np.flatnonzero(query_mask)
    bank_rows = np.flatnonzero(bank_mask)
    left: list[int] = []
    right: list[int] = []
    distance: list[float] = []
    for patch_id in np.intersect1d(rows["patch"][query_rows], rows["patch"][bank_rows]):
        q = query_rows[rows["patch"][query_rows] == patch_id]
        b = bank_rows[rows["patch"][bank_rows] == patch_id]
        if len(q) == 0 or len(b) == 0:
            continue
        tree = cKDTree(xy[rows["point"][b]])
        neighbor_count = min(2, len(b))
        d, index = tree.query(xy[rows["point"][q]], k=neighbor_count)
        if neighbor_count == 1:
            d = d[:, None]
            index = index[:, None]
        for query_row, distances, indices in zip(q, d, index):
            for candidate_distance, candidate_index in zip(distances, indices):
                bank_row = b[int(candidate_index)]
                if rows["point"][query_row] == rows["point"][bank_row]:
                    continue
                if candidate_distance <= MAX_PAIR_DISTANCE:
                    left.append(int(query_row))
                    right.append(int(bank_row))
                    distance.append(float(candidate_distance))
                break
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(distance, dtype=np.float32),
    )


def pair_values(
    rows: dict[str, np.ndarray], left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_gain = rows["gain"][left].reshape(len(left), -1)
    right_gain = rows["gain"][right].reshape(len(right), -1)
    left_coherence = rows["coherence"][left].reshape(len(left), -1)
    right_coherence = rows["coherence"][right].reshape(len(right), -1)
    phase_difference = np.angle(left_gain * np.conj(right_gain))
    coherence = np.minimum(left_coherence, right_coherence)
    amplitude = np.sqrt(np.abs(left_gain) * np.abs(right_gain))
    mass = np.sqrt(rows["mass"][left] * rows["mass"][right])[:, None]
    weight = mass * coherence * amplitude
    weight[coherence < MIN_CENTER_COHERENCE] = 0.0
    length_difference = rows["length"][left] - rows["length"][right]
    return phase_difference.astype(np.float64), length_difference.astype(np.float64), weight


def fit_beta(phase: np.ndarray, delta_length: np.ndarray, weight: np.ndarray) -> float:
    flat_phase = phase.reshape(-1)
    flat_length = np.repeat(delta_length, phase.shape[1])
    flat_weight = weight.reshape(-1).astype(np.float64)
    valid = flat_weight > 0.0
    flat_phase = flat_phase[valid]
    flat_length = flat_length[valid]
    flat_weight = flat_weight[valid]
    score = np.empty(len(BETA_GRID), dtype=np.float64)
    for start in range(0, len(BETA_GRID), 256):
        beta = BETA_GRID[start : start + 256]
        score[start : start + len(beta)] = np.sum(
            flat_weight[:, None]
            * np.cos(flat_phase[:, None] + flat_length[:, None] * beta[None, :]),
            axis=0,
        ) / np.maximum(np.sum(flat_weight), 1e-30)
    return float(BETA_GRID[int(np.argmax(score))])


def weighted_metrics(
    phase: np.ndarray,
    delta_length: np.ndarray,
    weight: np.ndarray,
    beta: float,
) -> dict[str, float]:
    residual = np.angle(np.exp(1j * (phase + beta * delta_length[:, None])))
    denominator = max(float(np.sum(weight)), 1e-30)
    return {
        "cosine": float(np.sum(weight * np.cos(residual)) / denominator),
        "phase_mae": float(np.sum(weight * np.abs(residual)) / denominator),
        "valid_channel_pairs": int(np.sum(weight > 0.0)),
    }


def main() -> None:
    for path in (v297.COMPONENT_CACHE, v305.MATCH_CACHE, CANDIDATE_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    gain_data = recover_center_gains()
    rows = matched_rows(gain_data)
    train_xy = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    valid_points = np.unique(rows["point"])
    folds = v315.disjoint_folds(valid_points, train_xy)
    rng = np.random.default_rng(325)
    fold_rows = []

    for fold, query_points in enumerate(folds):
        fit_points = np.setdiff1d(valid_points, query_points)
        fit_left, fit_right, fit_distance = nearest_patch_pairs(
            rows, fit_points, fit_points, train_xy
        )
        query_left, query_right, query_distance = nearest_patch_pairs(
            rows, query_points, fit_points, train_xy
        )
        fit_phase, fit_length, fit_weight = pair_values(rows, fit_left, fit_right)
        query_phase, query_length, query_weight = pair_values(
            rows, query_left, query_right
        )
        beta = fit_beta(fit_phase, fit_length, fit_weight)
        raw = weighted_metrics(query_phase, query_length, query_weight, 0.0)
        physical = weighted_metrics(query_phase, query_length, query_weight, beta)
        shuffled_cosine = []
        for _ in range(SHUFFLE_REPEATS):
            shuffled = rng.permutation(query_length)
            shuffled_cosine.append(
                weighted_metrics(query_phase, shuffled, query_weight, beta)["cosine"]
            )
        fold_rows.append(
            {
                "fold": fold,
                "fit_point_count": int(len(fit_points)),
                "query_point_count": int(len(query_points)),
                "fit_pair_count": int(len(fit_left)),
                "query_pair_count": int(len(query_left)),
                "fit_pair_distance": summarize(fit_distance),
                "query_pair_distance": summarize(query_distance),
                "selected_beta_rad_per_m": beta,
                "selected_frequency_ghz": beta * 299_792_458.0 / (2.0 * np.pi) / 1e9,
                "raw": raw,
                "physical": physical,
                "cosine_gain": physical["cosine"] - raw["cosine"],
                "phase_mae_reduction": (
                    raw["phase_mae"] - physical["phase_mae"]
                ) / max(raw["phase_mae"], 1e-30),
                "shuffled_length_cosine": summarize(np.asarray(shuffled_cosine)),
                "physical_minus_shuffle": physical["cosine"]
                - float(np.mean(shuffled_cosine)),
            }
        )
        print(
            f"v325 fold={fold} beta={beta:.3f} raw={raw['cosine']:.4f} "
            f"physical={physical['cosine']:.4f} pairs={len(query_left)}",
            flush=True,
        )

    cosine_gain = float(np.mean([row["cosine_gain"] for row in fold_rows]))
    mae_reduction = float(np.mean([row["phase_mae_reduction"] for row in fold_rows]))
    physical_minus_shuffle = float(
        np.mean([row["physical_minus_shuffle"] for row in fold_rows])
    )
    fold_wins = int(sum(row["cosine_gain"] > 0.0 for row in fold_rows))
    passed = (
        cosine_gain >= 0.15
        and mae_reduction >= 0.15
        and physical_minus_shuffle >= 0.10
        and fold_wins >= 4
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "For the same official-pointcloud wall-path family, removing the path-specific "
            "carrier phase beta*length makes official-H complex gains predictable across UEs."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H continuous-path complex gains",
                "official train coordinates",
                "official-pointcloud one-bounce path identity and length",
            ],
            "does_not_use": [
                "test truth or test channel",
                "synthetic channel labels or teacher outputs",
                "RT path coefficients or restored scene materials",
            ],
        },
        "parameters": {
            "beta_grid_rad_per_m": [float(BETA_GRID[0]), float(BETA_GRID[-1]), 0.02],
            "max_pair_distance_m": MAX_PAIR_DISTANCE,
            "minimum_center_coherence": MIN_CENTER_COHERENCE,
            "shuffle_repeats": SHUFFLE_REPEATS,
            "matched_row_count": int(len(rows["point"])),
            "matched_point_count": int(len(valid_points)),
            "matched_patch_count": int(len(np.unique(rows["patch"]))),
        },
        "fold_rows": fold_rows,
        "summary": {
            "cosine_gain": cosine_gain,
            "phase_mae_reduction": mae_reduction,
            "physical_minus_shuffle": physical_minus_shuffle,
            "fold_wins": fold_wins,
        },
        "decision_rule": {
            "cosine_gain": ">=0.15",
            "phase_mae_reduction": ">=0.15",
            "physical_minus_shuffle": ">=0.10",
            "fold_wins": ">=4/5",
            "passed": bool(passed),
        },
        "outputs": {
            "center_gains": PATHS.relative(CENTER_GAIN_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
