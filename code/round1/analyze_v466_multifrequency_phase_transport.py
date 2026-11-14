#!/usr/bin/env python3
"""Audit multi-frequency Sionna path events for phase-only transport.

This is an analysis-only experiment. Same-coordinate Sionna channels are used
to test whether a path event shared by 6.0 and 6.7 GHz can identify official-H
reflection support and release the carrier-phase law found in v325/v327.

The deployed probe changes only CIR phase. Every link's delay-domain magnitude,
link power, PDP, and point power remain exactly those of the selected carrier.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
import analyze_v326_complex_path_subspace_transport as v326
import analyze_v328_patch_existence_support_field as v328
import analyze_v454_v453_official_fullh_local_score as v454
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 466
VERSION = "v466_multifrequency_phase_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
JOINT_CACHE = OUT_DIR / "v466_multifrequency_joint_sh.npy"
EVIDENCE_CACHE = OUT_DIR / "v466_candidate_evidence.npz"

SIONNA_60 = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
SIONNA_67 = (
    PATHS.cache_dir(205)
    / "benchmark_sector_m30_track_d5_f6p7"
    / "train_clean_H.npy"
)
V98_BASE_AD = PATHS.shared_features / "train_v98like_loo_base_ad.npy"
V140_BASE_AD = PATHS.cache_dir(156) / "train_v140eq_crossfit_base_ad.npy"

N_DELAY = 192
N_H = 16
JOINT_BATCH = 4
DELAY_RADIUS = 3
H_RADIUS = 1
AGREEMENT_SCALE = 0.75
PRECISION_TARGET = 0.80
INNER_COUNT = 240
ALPHA_GRID = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
MIN_SCORE_DELTA = 0.005
MIN_FOLD_WINS = 4
PDP_LOCK_TOLERANCE = 2e-6


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def circular_signed_delta(left: np.ndarray, right: np.ndarray, period: float) -> np.ndarray:
    return (np.asarray(left) - np.asarray(right) + period / 2.0) % period - period / 2.0


def circular_distance(left: np.ndarray, right: np.ndarray, period: float) -> np.ndarray:
    return np.abs(circular_signed_delta(left, right, period))


def build_joint_cache() -> np.ndarray:
    if JOINT_CACHE.is_file():
        value = np.load(JOINT_CACHE, mmap_mode="r")
        if value.shape != (2, 2000, N_DELAY, N_H):
            raise ValueError(f"Unexpected v466 joint cache shape: {value.shape}")
        return value

    output = np.lib.format.open_memmap(
        JOINT_CACHE,
        mode="w+",
        dtype=np.float32,
        shape=(2, 2000, N_DELAY, N_H),
    )
    for source_index, source_path in enumerate((SIONNA_60, SIONNA_67)):
        channel = np.load(source_path, mmap_mode="r")
        for start in range(0, len(channel), JOINT_BATCH):
            end = min(start + JOINT_BATCH, len(channel))
            block = np.asarray(channel[start:end], dtype=np.complex64)
            cir = np.fft.ifft(block, axis=-1).reshape(
                end - start, 2, 16, 8, 4, N_DELAY
            )
            angle_delay = np.fft.fft(cir, axis=2)
            power = np.sum(
                np.abs(angle_delay).astype(np.float32) ** 2,
                axis=(1, 3, 4),
                dtype=np.float64,
            ).transpose(0, 2, 1)
            power = power.astype(np.float32)
            power /= np.maximum(
                np.sum(power, axis=(1, 2), keepdims=True, dtype=np.float64),
                1e-30,
            ).astype(np.float32)
            output[source_index, start:end] = power
            if end % 200 == 0 or end == len(channel):
                print(
                    f"v466 joint source={source_index} {end}/{len(channel)}",
                    flush=True,
                )
        output.flush()
    return np.load(JOINT_CACHE, mmap_mode="r")


def refine_peak(profile: np.ndarray, indices: np.ndarray, best: int) -> float:
    center = float(profile[best])
    left = float(profile[max(best - 1, 0)])
    right = float(profile[min(best + 1, len(profile) - 1)])
    log_left = np.log(max(left, 1e-30))
    log_center = np.log(max(center, 1e-30))
    log_right = np.log(max(right, 1e-30))
    denominator = log_left - 2.0 * log_center + log_right
    offset = 0.0
    if best > 0 and best + 1 < len(profile) and abs(denominator) > 1e-12:
        offset = float(np.clip(0.5 * (log_left - log_right) / denominator, -0.5, 0.5))
    return float((int(indices[best]) + offset) % N_DELAY)


def sample_candidate_source(
    joint: np.ndarray,
    geometry_delay: np.ndarray,
    h_center: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.zeros(valid.shape, dtype=np.float32)
    peak = np.full(valid.shape, np.nan, dtype=np.float32)
    for point_id in range(len(valid)):
        for candidate_id in np.flatnonzero(valid[point_id]):
            center_d = int(np.rint(geometry_delay[point_id, candidate_id]))
            delay_index = (
                center_d + np.arange(-DELAY_RADIUS, DELAY_RADIUS + 1)
            ) % N_DELAY
            center_h = int(np.rint(h_center[point_id, candidate_id]))
            h_index = (center_h + np.arange(-H_RADIUS, H_RADIUS + 1)) % N_H
            local = joint[point_id][np.ix_(delay_index, h_index)]
            profile = np.sum(local, axis=1, dtype=np.float64).astype(np.float32)
            best = int(np.argmax(profile))
            score[point_id, candidate_id] = float(profile[best])
            peak[point_id, candidate_id] = refine_peak(profile, delay_index, best)
        if (point_id + 1) % 250 == 0 or point_id + 1 == len(valid):
            print(f"v466 candidate evidence {point_id + 1}/{len(valid)}", flush=True)
    return score, peak


def build_candidate_evidence(
    candidate: dict[str, np.ndarray], valid: np.ndarray
) -> dict[str, np.ndarray]:
    if EVIDENCE_CACHE.is_file():
        data = np.load(EVIDENCE_CACHE)
        return {key: data[key] for key in data.files}

    joint = build_joint_cache()
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    geometry_delay = (
        float(calibration["intercept_bins"])
        + float(calibration["slope_bins_per_meter"])
        * candidate["path_length"].astype(np.float32)
    ) % N_DELAY
    h_center = v305.expected_h_bin(candidate["reflection_point"]).astype(np.float32)
    score_60, peak_60 = sample_candidate_source(
        joint[0], geometry_delay, h_center, valid
    )
    score_67, peak_67 = sample_candidate_source(
        joint[1], geometry_delay, h_center, valid
    )
    agreement = circular_distance(peak_60, peak_67, N_DELAY)
    stability = np.exp(-0.5 * (agreement / AGREEMENT_SCALE) ** 2).astype(np.float32)
    score = np.sqrt(np.maximum(score_60 * score_67, 0.0)).astype(np.float32)
    score *= np.nan_to_num(stability, nan=0.0)
    delta = circular_signed_delta(peak_67, peak_60, N_DELAY)
    peak = (peak_60 + 0.5 * delta) % N_DELAY
    peak[~valid] = np.nan
    score[~valid] = 0.0
    np.savez_compressed(
        EVIDENCE_CACHE,
        score=score,
        peak_delay=peak.astype(np.float32),
        score_60=score_60,
        score_67=score_67,
        peak_60=peak_60,
        peak_67=peak_67,
        delay_agreement=agreement.astype(np.float32),
        geometry_delay=geometry_delay.astype(np.float32),
    )
    data = np.load(EVIDENCE_CACHE)
    return {key: data[key] for key in data.files}


def threshold_metrics(
    score: np.ndarray,
    label: np.ndarray,
    valid: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
) -> tuple[float, dict[str, float], dict[str, float]]:
    fit_mask = valid[fit_ids]
    selected = v328.precision_recall_at_target(
        score[fit_ids][fit_mask], label[fit_ids][fit_mask], PRECISION_TARGET
    )
    threshold = float(selected["score_threshold"])
    query_mask = valid[query_ids]
    query_score = score[query_ids][query_mask]
    query_label = label[query_ids][query_mask]
    binary = query_label > 0.0
    ap = float(average_precision_score(binary, query_score))
    chosen = query_score >= threshold
    true_chosen = chosen & binary
    query_metrics = {
        "average_precision": ap,
        "selected_count": int(np.sum(chosen)),
        "count_precision": float(np.sum(true_chosen) / max(np.sum(chosen), 1)),
        "count_recall": float(np.sum(true_chosen) / max(np.sum(binary), 1)),
        "mass_recall": float(
            np.sum(query_label[true_chosen], dtype=np.float64)
            / max(np.sum(query_label[binary], dtype=np.float64), 1e-30)
        ),
    }
    return threshold, selected, query_metrics


def weighted_median(value: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(value)
    sorted_value = value[order]
    sorted_weight = weight[order]
    index = int(np.searchsorted(np.cumsum(sorted_weight), 0.5 * np.sum(sorted_weight)))
    return float(sorted_value[min(index, len(sorted_value) - 1)])


def fit_delay_offset(
    rows: dict[str, np.ndarray],
    observed_delay: np.ndarray,
    evidence: dict[str, np.ndarray],
    fit_ids: np.ndarray,
) -> float:
    mask = np.isin(rows["point"], fit_ids)
    point = rows["point"][mask]
    candidate_id = rows["candidate"][mask]
    synthetic = evidence["peak_delay"][point, candidate_id]
    truth = observed_delay[mask]
    valid = np.isfinite(synthetic)
    residual = circular_signed_delta(truth[valid], synthetic[valid], N_DELAY)
    weight = (
        rows["mass"][mask][valid]
        * np.maximum(evidence["score"][point[valid], candidate_id[valid]], 1e-8)
    ).astype(np.float64)
    return weighted_median(residual.astype(np.float64), weight)


def delay_error(
    rows: dict[str, np.ndarray],
    observed_delay: np.ndarray,
    evidence: dict[str, np.ndarray],
    query_ids: np.ndarray,
    offset: float,
) -> dict[str, float | int]:
    mask = np.isin(rows["point"], query_ids)
    point = rows["point"][mask]
    candidate_id = rows["candidate"][mask]
    prediction = (evidence["peak_delay"][point, candidate_id] + offset) % N_DELAY
    truth = observed_delay[mask]
    error = circular_distance(prediction, truth, N_DELAY)
    valid = np.isfinite(error)
    return summarize(error[valid])


def render_path(gain: np.ndarray, delay: float) -> np.ndarray:
    frequency = np.arange(N_DELAY, dtype=np.float32)
    steering = np.exp(-2j * np.pi * frequency * float(delay) / N_DELAY).astype(
        np.complex64
    )
    return (gain.reshape(-1, 1) * steering[None, :]).reshape(256, 4, N_DELAY)


def row_lookup(rows: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    return {
        (int(point_id), int(patch_id)): row
        for row, (point_id, patch_id) in enumerate(zip(rows["point"], rows["patch"]))
    }


def pathwise_prediction(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    train_pos: np.ndarray,
    channel: np.ndarray,
    candidate: dict[str, np.ndarray],
    evidence: dict[str, np.ndarray],
    threshold: float,
    delay_offset: float,
    rows: dict[str, np.ndarray],
    gain: np.ndarray,
    observed_delay: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    tree = cKDTree(train_pos[fit_ids, :2])
    distance, local = tree.query(
        train_pos[query_ids, :2], k=min(v326.K_BASE, len(fit_ids))
    )
    if distance.ndim == 1:
        distance = distance[:, None]
        local = local[:, None]
    neighbor_ids = fit_ids[local]
    weight = 1.0 / np.maximum(distance, 0.05) ** v326.BASE_POWER
    weight /= weight.sum(axis=1, keepdims=True)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    radial_distance = np.linalg.norm(train_pos - bs[None, :], axis=1)
    lookup = row_lookup(rows)
    base_output = np.empty(
        (len(query_ids), channel.shape[1], channel.shape[2], channel.shape[3]),
        dtype=np.complex64,
    )
    corrected_output = np.empty_like(base_output)
    corrected_paths = np.zeros(len(query_ids), dtype=np.float32)
    supporting_neighbors = np.zeros(len(query_ids), dtype=np.float32)

    for query_row, query_id in enumerate(query_ids):
        neighbors = neighbor_ids[query_row]
        radial_phase = np.exp(
            1j
            * v326.BASE_BETA
            * (radial_distance[neighbors] - radial_distance[query_id])
        ).astype(np.complex64)
        coefficient = (weight[query_row] * radial_phase).astype(np.complex64)
        neighbor_h = np.asarray(channel[neighbors], dtype=np.complex64)
        base = np.sum(
            coefficient[:, None, None, None] * neighbor_h,
            axis=0,
            dtype=np.complex64,
        )
        prediction = base.copy()
        active = np.flatnonzero(
            (candidate["score"][query_id] > 0.0)
            & (evidence["score"][query_id] >= threshold)
            & np.isfinite(evidence["peak_delay"][query_id])
        )
        if len(active) > v326.MAX_PATHS:
            active = active[
                np.argsort(evidence["score"][query_id, active])[::-1][
                    : v326.MAX_PATHS
                ]
            ]
        used_patch: set[int] = set()
        for candidate_id in active:
            patch_id = int(candidate["patch_id"][query_id, candidate_id])
            if patch_id in used_patch:
                continue
            support = [
                (neighbor_row, lookup[(int(point_id), patch_id)])
                for neighbor_row, point_id in enumerate(neighbors)
                if (int(point_id), patch_id) in lookup
            ]
            if not support:
                continue
            used_patch.add(patch_id)
            query_length = float(candidate["path_length"][query_id, candidate_id])
            query_delay = float(
                (evidence["peak_delay"][query_id, candidate_id] + delay_offset)
                % N_DELAY
            )
            for neighbor_row, bank_row in support:
                bank_gain = gain[bank_row]
                bank_path = render_path(bank_gain, float(observed_delay[bank_row]))
                prediction -= coefficient[neighbor_row] * bank_path
                transported_gain = bank_gain * np.complex64(
                    np.exp(
                        -1j
                        * beta
                        * (query_length - float(rows["length"][bank_row]))
                    )
                )
                query_path = render_path(transported_gain, query_delay)
                prediction += np.float32(weight[query_row, neighbor_row]) * query_path
                supporting_neighbors[query_row] += 1.0
            corrected_paths[query_row] += 1.0
        base_output[query_row] = base
        corrected_output[query_row] = prediction

    return base_output, corrected_output, {
        "corrected_paths": summarize(corrected_paths),
        "supporting_neighbors": summarize(supporting_neighbors),
        "active_fraction": float(np.mean(corrected_paths > 0.0)),
    }


def phase_delta(base: np.ndarray, corrected: np.ndarray) -> np.ndarray:
    base_cir = np.fft.ifft(base, axis=-1)
    corrected_cir = np.fft.ifft(corrected, axis=-1)
    return np.angle(corrected_cir * np.conj(base_cir)).astype(np.float32)


def phase_only(carrier: np.ndarray, delta: np.ndarray, alpha: float) -> np.ndarray:
    cir = np.fft.ifft(carrier, axis=-1)
    output = cir * np.exp(1j * float(alpha) * delta).astype(np.complex64)
    return np.fft.fft(output, axis=-1).astype(np.complex64)


def load_carrier(path, ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    base_ad = np.load(path, mmap_mode="r")
    return v454.restore_raw_h(
        np.asarray(base_ad[ids], dtype=np.complex64), positions[ids]
    )


def choose_alpha(
    carrier: np.ndarray, delta: np.ndarray, truth: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        metrics = v326.score(phase_only(carrier, delta, float(alpha)), truth)
        rows.append({"alpha": float(alpha), **metrics})
    best = max(rows, key=lambda row: row["score"])
    return float(best["alpha"]), rows


def evaluate_carrier(
    carrier: np.ndarray,
    delta: np.ndarray,
    alpha: float,
    truth: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    base_metrics = v326.score(carrier, truth)
    prediction = phase_only(carrier, delta, alpha)
    metrics = v326.score(prediction, truth)
    return base_metrics, metrics, {
        "score_delta": metrics["score"] - base_metrics["score"],
        "pas_delta": metrics["pas"] - base_metrics["pas"],
        "pdp_delta": metrics["pdp"] - base_metrics["pdp"],
        "inv_nmse_delta": metrics["inv_nmse"] - base_metrics["inv_nmse"],
    }


def summarize_carrier(fold_rows: list[dict[str, object]], name: str) -> dict[str, float | int]:
    deltas = [row["carriers"][name]["delta"] for row in fold_rows]
    return {
        key: float(np.mean([row[key] for row in deltas]))
        for key in ("score_delta", "pas_delta", "pdp_delta", "inv_nmse_delta")
    } | {"fold_wins": int(sum(row["score_delta"] > 0.0 for row in deltas))}


def inactive_diagnostics(count: int) -> dict[str, object]:
    zeros = np.zeros(count, dtype=np.float32)
    return {
        "corrected_paths": summarize(zeros),
        "supporting_neighbors": summarize(zeros),
        "active_fraction": 0.0,
    }


def inactive_carrier_row() -> dict[str, object]:
    return {
        "selected_alpha": 0.0,
        "inner_alpha_rows": [],
        "base": None,
        "prediction": None,
        "delta": {
            "score_delta": 0.0,
            "pas_delta": 0.0,
            "pdp_delta": 0.0,
            "inv_nmse_delta": 0.0,
        },
    }


def main() -> None:
    required = (
        SIONNA_60,
        SIONNA_67,
        V98_BASE_AD,
        V140_BASE_AD,
        v297.COMPONENT_CACHE,
        v305.MATCH_CACHE,
        v306.TRAIN_CANDIDATES,
        v325.CENTER_GAIN_CACHE,
        v326.GAIN_CACHE,
    )
    for path in required:
        require_file(path)

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    candidate_data = np.load(v306.TRAIN_CANDIDATES)
    candidate = {key: candidate_data[key] for key in candidate_data.files}
    candidate_valid, label = v328.build_labels(candidate)
    evidence = build_candidate_evidence(candidate, candidate_valid)
    gain_data = v325.recover_center_gains()
    rows = v325.matched_rows(gain_data)
    gain = v326.recover_matched_gain_tensors(rows)
    component = np.load(v297.COMPONENT_CACHE)
    observed_delay = component["delay"][rows["point"], rows["component"]].astype(
        np.float32
    )
    valid_ids = np.flatnonzero(component["valid"].any(axis=1)).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    fold_rows: list[dict[str, object]] = []

    for fold, outer_ids in enumerate(folds):
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        rng = np.random.default_rng(46_600 + fold)
        inner_ids = np.sort(
            rng.choice(outer_fit, size=min(INNER_COUNT, len(outer_fit) // 4), replace=False)
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)

        inner_threshold, inner_fit_precision, _inner_query_precision = threshold_metrics(
            evidence["score"], label, candidate_valid, inner_fit, inner_ids
        )
        inner_offset = fit_delay_offset(rows, observed_delay, evidence, inner_fit)
        inner_beta = v326.estimate_beta(rows, inner_fit, train_pos[:, :2])
        selected_alpha: dict[str, float] = {}
        alpha_rows: dict[str, list[dict[str, float]]] = {}
        if (
            np.isfinite(inner_threshold)
            and inner_fit_precision["selected_count"] > 0
            and _inner_query_precision["selected_count"] > 0
        ):
            inner_radial, inner_corrected, inner_diagnostics = pathwise_prediction(
                inner_ids,
                inner_fit,
                train_pos,
                channel,
                candidate,
                evidence,
                inner_threshold,
                inner_offset,
                rows,
                gain,
                observed_delay,
                inner_beta,
            )
            inner_delta = phase_delta(inner_radial, inner_corrected)
            inner_truth = np.asarray(channel[inner_ids], dtype=np.complex64)
            inner_carriers = {
                "radial": inner_radial,
                "v98": load_carrier(V98_BASE_AD, inner_ids, train_pos),
                "v140": load_carrier(V140_BASE_AD, inner_ids, train_pos),
            }
            for name, carrier in inner_carriers.items():
                selected_alpha[name], alpha_rows[name] = choose_alpha(
                    carrier, inner_delta, inner_truth
                )
        else:
            inner_diagnostics = inactive_diagnostics(len(inner_ids))
            for name in ("radial", "v98", "v140"):
                selected_alpha[name] = 0.0
                alpha_rows[name] = []

        outer_threshold, outer_fit_precision, outer_query_precision = threshold_metrics(
            evidence["score"], label, candidate_valid, outer_fit, outer_ids
        )
        outer_offset = fit_delay_offset(rows, observed_delay, evidence, outer_fit)
        outer_beta = v326.estimate_beta(rows, outer_fit, train_pos[:, :2])
        carrier_rows: dict[str, object] = {}
        if (
            np.isfinite(outer_threshold)
            and outer_fit_precision["selected_count"] > 0
            and outer_query_precision["selected_count"] > 0
            and any(value > 0.0 for value in selected_alpha.values())
        ):
            outer_radial, outer_corrected, outer_diagnostics = pathwise_prediction(
                outer_ids,
                outer_fit,
                train_pos,
                channel,
                candidate,
                evidence,
                outer_threshold,
                outer_offset,
                rows,
                gain,
                observed_delay,
                outer_beta,
            )
            outer_delta = phase_delta(outer_radial, outer_corrected)
            outer_truth = np.asarray(channel[outer_ids], dtype=np.complex64)
            outer_carriers = {
                "radial": outer_radial,
                "v98": load_carrier(V98_BASE_AD, outer_ids, train_pos),
                "v140": load_carrier(V140_BASE_AD, outer_ids, train_pos),
            }
            for name, carrier in outer_carriers.items():
                if selected_alpha[name] <= 0.0:
                    carrier_rows[name] = inactive_carrier_row()
                    continue
                base_metrics, metrics, delta_metrics = evaluate_carrier(
                    carrier, outer_delta, selected_alpha[name], outer_truth
                )
                carrier_rows[name] = {
                    "selected_alpha": selected_alpha[name],
                    "inner_alpha_rows": alpha_rows[name],
                    "base": base_metrics,
                    "prediction": metrics,
                    "delta": delta_metrics,
                }
        else:
            outer_diagnostics = inactive_diagnostics(len(outer_ids))
            for name in ("radial", "v98", "v140"):
                carrier_rows[name] = inactive_carrier_row()

        fold_rows.append(
            {
                "fold": fold,
                "outer_count": int(len(outer_ids)),
                "inner_count": int(len(inner_ids)),
                "inner_threshold": inner_threshold,
                "outer_threshold": outer_threshold,
                "inner_fit_precision": inner_fit_precision,
                "outer_fit_precision": outer_fit_precision,
                "outer_query_precision": outer_query_precision,
                "inner_delay_offset": inner_offset,
                "outer_delay_offset": outer_offset,
                "outer_delay_error": delay_error(
                    rows, observed_delay, evidence, outer_ids, outer_offset
                ),
                "inner_beta": inner_beta,
                "outer_beta": outer_beta,
                "inner_diagnostics": inner_diagnostics,
                "outer_diagnostics": outer_diagnostics,
                "carriers": carrier_rows,
            }
        )
        print(
            f"v466 fold={fold} precision={outer_query_precision['count_precision']:.3f} "
            f"recall={outer_query_precision['mass_recall']:.3f} "
            f"v98={carrier_rows['v98']['delta']['score_delta']:+.6f}",
            flush=True,
        )

    summary = {
        name: summarize_carrier(fold_rows, name) for name in ("radial", "v98", "v140")
    }
    support_summary = {
        "average_precision": float(
            np.mean([row["outer_query_precision"]["average_precision"] for row in fold_rows])
        ),
        "count_precision": float(
            np.mean([row["outer_query_precision"]["count_precision"] for row in fold_rows])
        ),
        "count_recall": float(
            np.mean([row["outer_query_precision"]["count_recall"] for row in fold_rows])
        ),
        "mass_recall": float(
            np.mean([row["outer_query_precision"]["mass_recall"] for row in fold_rows])
        ),
        "delay_mae_bins": float(
            np.mean([row["outer_delay_error"]["mean"] for row in fold_rows])
        ),
    }
    primary = summary["v98"]
    passed = (
        primary["score_delta"] >= MIN_SCORE_DELTA
        and primary["fold_wins"] >= MIN_FOLD_WINS
        and abs(primary["pdp_delta"]) <= PDP_LOCK_TOLERANCE
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A reflection event jointly present at 6.0 and 6.7 GHz supplies enough "
            "path-existence and delay identity to release official-H path carrier phase "
            "transport without changing amplitude, PDP, or power."
        ),
        "compliance_boundary": {
            "status": "noncompliant_same_coordinate_sionna_diagnostic",
            "reason": (
                "The event selector reads Sionna channels simulated at the same official "
                "train coordinates; this experiment is evidence only and cannot be a final model."
            ),
        },
        "parameters": {
            "frequencies_ghz": [6.0, 6.7],
            "candidate_delay_radius": DELAY_RADIUS,
            "candidate_h_radius": H_RADIUS,
            "frequency_delay_agreement_scale": AGREEMENT_SCALE,
            "precision_target": PRECISION_TARGET,
            "phase_alpha_grid": ALPHA_GRID.tolist(),
            "phase_law_beta": "fit from official-H matched paths inside each fold",
            "primary_carrier": "v98-like official OOF full-H",
        },
        "support_summary": support_summary,
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "primary_score_delta": f">={MIN_SCORE_DELTA}",
            "primary_fold_wins": f">={MIN_FOLD_WINS}/5",
            "absolute_pdp_delta": f"<={PDP_LOCK_TOLERANCE}",
            "passed": bool(passed),
        },
        "next_action": (
            "Build a test candidate only after this audit passes; otherwise stop phase release "
            "from multi-frequency support and return to a new information source."
        ),
        "outputs": {
            "report": PATHS.relative(REPORT),
            "joint_cache": PATHS.relative(JOINT_CACHE),
            "evidence_cache": PATHS.relative(EVIDENCE_CACHE),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"support": support_summary, "summary": summary, "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
