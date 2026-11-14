#!/usr/bin/env python3
"""Recover per-delay path gain tensors and test their identity fingerprints."""

from __future__ import annotations

import json

import numpy as np

import analyze_v254_route_topology_spectral_field as v254
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v299_component_correspondence_oracle as v299
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 300
VERSION = "v300_path_gain_fingerprint_oracle"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FINGERPRINT_CACHE = OUT_DIR / "v300_train_path_gain_fingerprints.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

OUTER_SEEDS = (300, 1300, 2300, 3300, 4300)
OUTER_COUNT = 320
MATCH_DELAY_THRESHOLD = 1.0
RIDGE = 1e-4


def normalize_last(value: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = out.sum(axis=-1, keepdims=True)
    return (out / np.maximum(total, 1e-20)).astype(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    return np.sum(x * y, axis=-1) / np.maximum(
        np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1), 1e-30
    )


def extract_fingerprints() -> dict[str, np.ndarray]:
    if FINGERPRINT_CACHE.is_file():
        data = np.load(FINGERPRINT_CACHE)
        return {key: data[key] for key in data.files}
    if not v297.COMPONENT_CACHE.is_file():
        raise FileNotFoundError(v297.COMPONENT_CACHE)
    component = np.load(v297.COMPONENT_CACHE)
    delays = component["delay"].astype(np.float64)
    valid = component["valid"].astype(bool)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    shape = delays.shape
    horizontal = np.zeros(shape + (16,), dtype=np.float32)
    vertical = np.zeros(shape + (8,), dtype=np.float32)
    polarization = np.zeros(shape + (2,), dtype=np.float32)
    ue_port = np.zeros(shape + (4,), dtype=np.float32)
    gain_power = np.zeros(shape, dtype=np.float32)
    h_ue_half_cosine = np.zeros(shape, dtype=np.float32)
    v_ue_half_cosine = np.zeros(shape, dtype=np.float32)
    reconstruction_nmse = np.ones(len(channel), dtype=np.float32)
    gram_condition = np.zeros(len(channel), dtype=np.float32)
    frequency = np.arange(v297.N_DELAY, dtype=np.float64)

    for index in range(len(channel)):
        active = np.flatnonzero(valid[index])
        if len(active) == 0:
            continue
        tau = delays[index, active]
        steering = np.exp(
            -2j * np.pi * frequency[:, None] * tau[None, :] / v297.N_DELAY
        ).astype(np.complex64)
        h = np.asarray(channel[index], dtype=np.complex64).reshape(-1, v297.N_DELAY)
        gram = steering.conj().T @ steering
        gram_condition[index] = float(np.linalg.cond(gram))
        regularized = gram + np.eye(len(active), dtype=np.complex64) * (
            RIDGE * np.trace(gram).real / max(len(active), 1)
        )
        gain = np.linalg.solve(regularized, steering.conj().T @ h.T)
        reconstructed = (steering @ gain).T
        reconstruction_nmse[index] = float(
            np.sum(np.abs(h - reconstructed) ** 2, dtype=np.float64)
            / max(float(np.sum(np.abs(h) ** 2, dtype=np.float64)), 1e-30)
        )
        tensor = gain.reshape(len(active), 2, 16, 8, 4)
        power = np.abs(tensor) ** 2
        h_spectrum_by_ue = np.abs(np.fft.fft(tensor, axis=2)) ** 2
        v_spectrum_by_ue = np.abs(np.fft.fft(tensor, axis=3)) ** 2
        h_spectrum = normalize_last(
            h_spectrum_by_ue.sum(axis=(1, 3, 4), dtype=np.float64)
        )
        v_spectrum = normalize_last(
            v_spectrum_by_ue.sum(axis=(1, 2, 4), dtype=np.float64)
        )
        pol_spectrum = normalize_last(power.sum(axis=(2, 3, 4), dtype=np.float64))
        ue_spectrum = normalize_last(power.sum(axis=(1, 2, 3), dtype=np.float64))
        h_first = normalize_last(
            h_spectrum_by_ue[..., :2].sum(axis=(1, 3, 4), dtype=np.float64)
        )
        h_second = normalize_last(
            h_spectrum_by_ue[..., 2:].sum(axis=(1, 3, 4), dtype=np.float64)
        )
        v_first = normalize_last(
            v_spectrum_by_ue[..., :2].sum(axis=(1, 2, 4), dtype=np.float64)
        )
        v_second = normalize_last(
            v_spectrum_by_ue[..., 2:].sum(axis=(1, 2, 4), dtype=np.float64)
        )
        horizontal[index, active] = h_spectrum
        vertical[index, active] = v_spectrum
        polarization[index, active] = pol_spectrum
        ue_port[index, active] = ue_spectrum
        gain_power[index, active] = power.sum(axis=(1, 2, 3, 4), dtype=np.float64)
        h_ue_half_cosine[index, active] = cosine(h_first, h_second)
        v_ue_half_cosine[index, active] = cosine(v_first, v_second)
        if (index + 1) % 200 == 0 or index + 1 == len(channel):
            print(f"v300 path gain fingerprints {index + 1}/{len(channel)}", flush=True)

    gain_power /= np.maximum(gain_power.sum(axis=1, keepdims=True), 1e-20)
    payload = {
        "horizontal": horizontal,
        "vertical": vertical,
        "polarization": polarization,
        "ue_port": ue_port,
        "gain_power": gain_power,
        "h_ue_half_cosine": h_ue_half_cosine,
        "v_ue_half_cosine": v_ue_half_cosine,
        "reconstruction_nmse": reconstruction_nmse,
        "gram_condition": gram_condition,
    }
    np.savez_compressed(FINGERPRINT_CACHE, **payload)
    return payload


def fingerprint_similarity(
    query_id: int,
    query_component: int,
    candidate_ids: np.ndarray,
    candidate_components: np.ndarray,
    fingerprint: dict[str, np.ndarray],
) -> np.ndarray:
    similarity_h = cosine(
        fingerprint["horizontal"][candidate_ids, candidate_components],
        fingerprint["horizontal"][query_id, query_component],
    )
    similarity_v = cosine(
        fingerprint["vertical"][candidate_ids, candidate_components],
        fingerprint["vertical"][query_id, query_component],
    )
    similarity_p = cosine(
        fingerprint["polarization"][candidate_ids, candidate_components],
        fingerprint["polarization"][query_id, query_component],
    )
    similarity_u = cosine(
        fingerprint["ue_port"][candidate_ids, candidate_components],
        fingerprint["ue_port"][query_id, query_component],
    )
    return (
        0.55 * similarity_h
        + 0.20 * similarity_v
        + 0.10 * similarity_p
        + 0.15 * similarity_u
    )


def analyze_matching(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    all_pos: np.ndarray,
    signatures: np.ndarray,
    blocker_count: np.ndarray,
    footprint: np.ndarray,
    z_max: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    delay: np.ndarray,
    component_mass: np.ndarray,
    valid: np.ndarray,
    fingerprint: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    raw = v274.build_raw_set(
        query_ids,
        bank_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )
    result: dict[str, list[float]] = {
        "fingerprint_weighted_mean_delay_error": [],
        "fingerprint_affine_delay_error": [],
        "selected_fingerprint_cosine": [],
        "per_neighbor_margin": [],
        "per_neighbor_candidate_count": [],
        "matched_neighbor_count": [],
    }
    for row, query_id in enumerate(query_ids):
        query_components = np.flatnonzero(valid[query_id])
        if len(query_components) == 0:
            continue
        query_mass = component_mass[query_id, query_components]
        query_mass /= np.maximum(np.sum(query_mass), 1e-20)
        neighbor_ids = raw.neighbor_ids[row]
        per_component: dict[str, list[tuple[float, float]]] = {
            key: [] for key in result
        }
        for query_component, path_mass in zip(query_components, query_mass):
            target_delay = float(delay[query_id, query_component])
            matched_delay = []
            matched_xy = []
            matched_weight = []
            matched_similarity = []
            margins = []
            candidate_counts = []
            for neighbor_row, neighbor_id in enumerate(neighbor_ids):
                candidates = np.flatnonzero(valid[neighbor_id])
                if len(candidates) == 0:
                    continue
                candidate_delay = delay[neighbor_id, candidates]
                distance = np.abs(v299.signed_circular_delta(candidate_delay, target_delay))
                keep = distance <= MATCH_DELAY_THRESHOLD
                if not np.any(keep):
                    continue
                candidates = candidates[keep]
                candidate_counts.append(float(len(candidates)))
                similarities = fingerprint_similarity(
                    int(query_id),
                    int(query_component),
                    np.full(len(candidates), int(neighbor_id), dtype=np.int64),
                    candidates,
                    fingerprint,
                )
                order = np.argsort(similarities)[::-1]
                selected = int(order[0])
                margin = float(
                    similarities[order[0]] - similarities[order[1]]
                    if len(order) > 1
                    else similarities[order[0]]
                )
                component = int(candidates[selected])
                matched_delay.append(float(delay[neighbor_id, component]))
                matched_xy.append(all_pos[neighbor_id, :2] - all_pos[query_id, :2])
                matched_similarity.append(float(similarities[selected]))
                matched_weight.append(
                    float(
                        raw.base_weights[row, neighbor_row]
                        * component_mass[neighbor_id, component]
                        * np.exp(3.0 * similarities[selected])
                    )
                )
                margins.append(margin)
            if not matched_delay:
                continue
            observed = np.asarray(matched_delay, dtype=np.float64)
            xy = np.asarray(matched_xy, dtype=np.float64)
            weight = np.asarray(matched_weight, dtype=np.float64)
            weight /= np.maximum(np.sum(weight), 1e-20)
            mean_prediction = (
                target_delay
                + float(
                    np.sum(weight * v299.signed_circular_delta(observed, target_delay))
                )
            ) % v297.N_DELAY
            mean_error = float(
                abs(v299.signed_circular_delta(mean_prediction, target_delay))
            )
            affine_prediction = v299.fit_affine_at_query(
                xy, observed, weight, target_delay
            ) if len(observed) >= 3 else mean_prediction
            affine_error = float(
                abs(v299.signed_circular_delta(affine_prediction, target_delay))
            )
            values = {
                "fingerprint_weighted_mean_delay_error": mean_error,
                "fingerprint_affine_delay_error": affine_error,
                "selected_fingerprint_cosine": float(np.average(matched_similarity, weights=weight)),
                "per_neighbor_margin": float(np.mean(margins)),
                "per_neighbor_candidate_count": float(np.mean(candidate_counts)),
                "matched_neighbor_count": float(len(observed)),
            }
            for key, value in values.items():
                per_component[key].append((float(path_mass), value))

        for key, values in per_component.items():
            if not values:
                continue
            weights = np.asarray([value[0] for value in values])
            numbers = np.asarray([value[1] for value in values])
            result[key].append(
                float(np.sum(weights * numbers) / np.maximum(np.sum(weights), 1e-20))
            )
    return {key: np.asarray(value, dtype=np.float32) for key, value in result.items()}


def main() -> None:
    fingerprint = extract_fingerprints()
    component = np.load(v297.COMPONENT_CACHE)
    delay = component["delay"].astype(np.float32)
    mass = component["weight"].astype(np.float32)
    valid = component["valid"].astype(bool)
    targets = np.load(v297.TARGET_CACHE)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    (
        all_pos,
        signatures,
        blocker_count,
        _scene_features,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = v274.load_geometry()

    combined: dict[str, list[np.ndarray]] = {}
    fold_rows = []
    for fold, seed in enumerate(OUTER_SEEDS):
        query_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        bank_ids = np.setdiff1d(valid_ids, query_ids)
        values = analyze_matching(
            query_ids,
            bank_ids,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
            delay,
            mass,
            valid,
            fingerprint,
        )
        summary = {key: summarize(value) for key, value in values.items()}
        fold_rows.append({"fold": fold, "seed": seed, "summary": summary})
        for key, value in values.items():
            combined.setdefault(key, []).append(value)
        print(
            f"v300 fold {fold + 1}/5 mean={summary['fingerprint_weighted_mean_delay_error']['mean']:.6f} "
            f"affine={summary['fingerprint_affine_delay_error']['mean']:.6f} "
            f"cos={summary['selected_fingerprint_cosine']['mean']:.6f}",
            flush=True,
        )

    path_weight = fingerprint["gain_power"] * valid
    overall = {key: summarize(np.concatenate(value)) for key, value in combined.items()}
    extraction = {
        "channel_reconstruction_nmse": summarize(fingerprint["reconstruction_nmse"][valid_ids]),
        "gram_condition": summarize(fingerprint["gram_condition"][valid_ids]),
        "horizontal_ue_half_cosine": summarize(
            fingerprint["h_ue_half_cosine"][valid]
        ),
        "vertical_ue_half_cosine": summarize(
            fingerprint["v_ue_half_cosine"][valid]
        ),
        "nnls_vs_gain_power_cosine": summarize(
            cosine(
                normalize_last(mass[valid_ids]),
                normalize_last(path_weight[valid_ids]),
            )
        ),
    }
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "diagnostic_truth_policy": (
            "Held-out query path fingerprints and delays seed oracle matching only; "
            "the experiment does not train or deploy a test predictor."
        ),
        "extraction": extraction,
        "matching": overall,
        "fold_rows": fold_rows,
        "outputs": {
            "fingerprints": PATHS.relative(FINGERPRINT_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(extraction, indent=2), flush=True)
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
