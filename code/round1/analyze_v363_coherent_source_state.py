#!/usr/bin/env python3
"""Strict carrier-dephased coherent virtual-source state experiment."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v311_effective_source_existence_boundary as v311
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
import analyze_v354_nested_virtual_source_transport as v354
import analyze_v359_fingerprint_subfamily_identifiability as v359
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 363
VERSION = "v363_coherent_source_state"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v363_coherent_source_state_oof.npz"

PAIR_NEIGHBORS = 2
MAX_PAIR_DISTANCE = 25.0
MIN_AP_LIFT = 0.05
MIN_MASS_COSINE_LIFT = 0.03
MIN_RENDERER_DELTA = 0.003
MIN_FOLD_WINS = 4


def fit_source_beta(
    assigned: v359.AssignedObservations,
    observation: v311.Observations,
    gain_data: dict[str, np.ndarray],
    train_xy: np.ndarray,
) -> tuple[float, dict[str, object]]:
    pairs: set[tuple[int, int]] = set()
    for source in np.unique(assigned.parent):
        rows = np.flatnonzero(assigned.parent == source)
        if len(rows) < 2:
            continue
        point_ids = observation.point_id[assigned.rows[rows]]
        take = min(PAIR_NEIGHBORS + 1, len(rows))
        distance, local = cKDTree(train_xy[point_ids]).query(
            train_xy[point_ids], k=take
        )
        if take == 1:
            continue
        for row_index, (row_distance, row_local) in enumerate(zip(distance, local)):
            added = 0
            for value, neighbor_index in zip(row_distance, row_local):
                if int(neighbor_index) == row_index or float(value) > MAX_PAIR_DISTANCE:
                    continue
                left = int(rows[row_index])
                right = int(rows[int(neighbor_index)])
                pairs.add((min(left, right), max(left, right)))
                added += 1
                if added >= PAIR_NEIGHBORS:
                    break
    if not pairs:
        raise RuntimeError("No fit-only same-source complex pairs for beta estimation")
    pair_array = np.asarray(sorted(pairs), dtype=np.int64)
    left = pair_array[:, 0]
    right = pair_array[:, 1]
    left_point = observation.point_id[assigned.rows[left]]
    right_point = observation.point_id[assigned.rows[right]]
    left_component = observation.component_id[assigned.rows[left]]
    right_component = observation.component_id[assigned.rows[right]]
    left_gain = gain_data["center_gain"][left_point, left_component].reshape(len(left), -1)
    right_gain = gain_data["center_gain"][right_point, right_component].reshape(len(right), -1)
    left_coherence = gain_data["center_coherence"][left_point, left_component].reshape(
        len(left), -1
    )
    right_coherence = gain_data["center_coherence"][right_point, right_component].reshape(
        len(right), -1
    )
    phase = np.angle(left_gain * np.conj(right_gain)).astype(np.float64)
    delta_length = (
        observation.length[assigned.rows[left]]
        - observation.length[assigned.rows[right]]
    ).astype(np.float64)
    coherence = np.minimum(left_coherence, right_coherence)
    amplitude = np.sqrt(np.abs(left_gain) * np.abs(right_gain))
    mass = np.sqrt(assigned.value[left] * assigned.value[right])[:, None]
    weight = mass * coherence * amplitude
    weight[coherence < v325.MIN_CENTER_COHERENCE] = 0.0
    if not np.any(weight > 0.0):
        raise RuntimeError("No coherent source channel pairs for beta estimation")
    beta = v325.fit_beta(phase, delta_length, weight)
    physical = v325.weighted_metrics(phase, delta_length, weight, beta)
    raw = v325.weighted_metrics(phase, delta_length, weight, 0.0)
    return beta, {
        "pair_count": int(len(pair_array)),
        "valid_channel_pairs": int(np.sum(weight > 0.0)),
        "raw": raw,
        "carrier_dephased": physical,
        "cosine_gain": float(physical["cosine"] - raw["cosine"]),
        "frequency_ghz": beta * 299_792_458.0 / (2.0 * np.pi) / 1e9,
    }


def build_complex_state(
    assigned: v359.AssignedObservations,
    observation: v311.Observations,
    gain_data: dict[str, np.ndarray],
    beta: float,
    source_count: int,
) -> tuple[np.ndarray, dict[str, float]]:
    state = np.zeros((assigned.target.shape[0], source_count, 8), dtype=np.complex64)
    point_id = observation.point_id[assigned.rows]
    component_id = observation.component_id[assigned.rows]
    gain = gain_data["center_gain"][point_id, component_id].reshape(len(point_id), -1)
    coherence = gain_data["center_coherence"][point_id, component_id].reshape(
        len(point_id), -1
    )
    weighted_gain = gain * np.sqrt(np.maximum(coherence, 0.0))
    norm = np.linalg.norm(weighted_gain, axis=1)
    fallback = norm < 1e-12
    weighted_gain[fallback] = gain[fallback]
    norm = np.linalg.norm(weighted_gain, axis=1)
    unresolved = norm < 1e-12
    weighted_gain[unresolved] = 1.0 + 0.0j
    norm = np.linalg.norm(weighted_gain, axis=1)
    direction = weighted_gain / np.maximum(norm[:, None], 1e-20)
    carrier = np.exp(1j * beta * observation.length[assigned.rows])[:, None]
    value = (
        direction
        * np.sqrt(np.maximum(assigned.value, 0.0))[:, None]
        * carrier
    ).astype(np.complex64)
    state[assigned.point_rows, assigned.parent] = value
    recovered = np.sum(np.abs(state) ** 2, axis=2)
    error = float(np.max(np.abs(recovered - assigned.target)))
    if not np.allclose(recovered, assigned.target, atol=1e-6):
        raise AssertionError("complex source state does not preserve hard source mass")
    return state, {
        "mass_conservation_max_error": error,
        "fallback_direction_count": int(np.sum(fallback)),
        "unresolved_direction_count": int(np.sum(unresolved)),
    }


def shuffled_phase_control(state: np.ndarray, seed: int) -> np.ndarray:
    output = state.copy()
    rng = np.random.default_rng(seed)
    active = np.linalg.norm(output, axis=2) > 0.0
    phase = rng.uniform(-np.pi, np.pi, size=active.shape).astype(np.float32)
    output *= np.exp(1j * phase)[:, :, None].astype(np.complex64)
    return output


def coherent_prediction(
    fit_state: np.ndarray,
    neighbor: v354.NeighborSet,
    base_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    prediction = np.sum(
        neighbor.weights[:, :, None, None] * fit_state[neighbor.ids],
        axis=1,
        dtype=np.complex128,
    ).astype(np.complex64)
    raw_mass = np.sum(np.abs(prediction) ** 2, axis=2).astype(np.float32)
    raw_total = raw_mass.sum(axis=1)
    base_total = base_mass.sum(axis=1)
    output = raw_mass.copy()
    valid = raw_total > 1e-12
    output[valid] *= (base_total[valid] / raw_total[valid])[:, None]
    output[~valid] = base_mass[~valid]
    max_error = float(np.max(np.abs(output.sum(axis=1) - base_total)))
    if not np.allclose(output.sum(axis=1), base_total, atol=1e-6):
        raise AssertionError("coherent prediction changed KNN total source mass")
    return output, prediction, {
        "raw_total_mass": summarize(raw_total),
        "base_total_mass": summarize(base_total),
        "mass_conservation_max_error": max_error,
        "fallback_query_count": int(np.sum(~valid)),
    }


def complex_cosine(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = prediction.reshape(len(prediction), -1).astype(np.complex128)
    truth = target.reshape(len(target), -1).astype(np.complex128)
    numerator = np.real(np.sum(pred * np.conj(truth), axis=1))
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    valid = denominator > 1e-20
    value = np.zeros(len(pred), dtype=np.float64)
    value[valid] = numerator[valid] / denominator[valid]
    return {
        "mean": float(np.mean(value[valid])) if np.any(valid) else 0.0,
        "valid_count": int(np.sum(valid)),
        "positive_fraction": float(np.mean(value[valid] > 0.0)) if np.any(valid) else 0.0,
    }


def metric_delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: float(left[key] - right[key]) for key in ("sh", "d", "weighted")}


def main() -> None:
    for path in (
        v354.TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v300.FINGERPRINT_CACHE,
        v325.CENTER_GAIN_CACHE,
        v354.TRAIN_CANDIDATES,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    target = np.load(v354.TARGET_CACHE)
    truth_sh = v354.normalize(target["sh"].astype(np.float32))
    truth_d = v354.normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    train_xy = train_pos[:, :2]
    observation, observation_report = v354.build_physical_observations()
    gain_data = v325.recover_center_gains()
    folds = v315.disjoint_folds(valid_ids, train_xy)

    fold_rows: list[dict[str, object]] = []
    oof_payload: dict[str, np.ndarray] = {}
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        centers, source_report = v311.discover_sources(observation, fit_ids)
        fit = v359.assign_observations(observation, fit_ids, centers, train_pos)
        outer = v359.assign_observations(observation, outer_ids, centers, train_pos)
        beta, beta_report = fit_source_beta(fit, observation, gain_data, train_xy)
        fit_state, fit_state_report = build_complex_state(
            fit, observation, gain_data, beta, len(centers)
        )
        outer_state, outer_state_report = build_complex_state(
            outer, observation, gain_data, beta, len(centers)
        )
        shuffled_state = shuffled_phase_control(
            fit_state, VERSION_NUMBER * 10_000 + fold
        )
        neighbor = v354.neighbors(outer_ids, fit_ids, train_xy, False)
        if np.intersect1d(np.unique(neighbor.ids), outer_ids).size:
            raise AssertionError("outer H entered v363 KNN current")
        fit_lookup = np.full(len(train_pos), -1, dtype=np.int64)
        fit_lookup[fit_ids] = np.arange(len(fit_ids), dtype=np.int64)
        local_neighbor = v354.NeighborSet(
            ids=fit_lookup[neighbor.ids], weights=neighbor.weights
        )
        if np.any(local_neighbor.ids < 0):
            raise AssertionError("v363 donor lookup failed")
        base_mass = np.sum(
            neighbor.weights[:, :, None] * fit.target[local_neighbor.ids],
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)
        coherent_mass, coherent_state, coherent_report = coherent_prediction(
            fit_state, local_neighbor, base_mass
        )
        shuffled_mass, shuffled_prediction_state, shuffled_report = coherent_prediction(
            shuffled_state, local_neighbor, base_mass
        )
        source_metrics = {
            "mass_knn": v311.evaluate_prediction(base_mass, outer.target),
            "coherent": v311.evaluate_prediction(coherent_mass, outer.target),
            "phase_shuffled": v311.evaluate_prediction(shuffled_mass, outer.target),
        }
        ap_lift = source_metrics["coherent"]["pair_ap"] - source_metrics["mass_knn"]["pair_ap"]
        mass_lift = (
            source_metrics["coherent"]["mass_cosine"]
            - source_metrics["mass_knn"]["mass_cosine"]
        )
        control_ap = source_metrics["coherent"]["pair_ap"] - source_metrics["phase_shuffled"]["pair_ap"]
        control_mass = (
            source_metrics["coherent"]["mass_cosine"]
            - source_metrics["phase_shuffled"]["mass_cosine"]
        )
        anchor = v354.anchor_from_neighbors(truth_sh, neighbor)
        knn_prediction, knn_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit.target,
            centers,
            train_xy,
        )
        coherent_channel, coherent_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit.target,
            centers,
            train_xy,
            query_mass_override=coherent_mass,
        )
        shuffled_channel, shuffled_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit.target,
            centers,
            train_xy,
            query_mass_override=shuffled_mass,
        )
        renderer_metrics = {
            "mass_knn": v354.metrics(
                knn_prediction, truth_sh[outer_ids], truth_d[outer_ids]
            ),
            "coherent": v354.metrics(
                coherent_channel, truth_sh[outer_ids], truth_d[outer_ids]
            ),
            "phase_shuffled": v354.metrics(
                shuffled_channel, truth_sh[outer_ids], truth_d[outer_ids]
            ),
        }
        renderer_delta = metric_delta(
            renderer_metrics["coherent"], renderer_metrics["mass_knn"]
        )
        renderer_control = metric_delta(
            renderer_metrics["coherent"], renderer_metrics["phase_shuffled"]
        )
        complex_metric = {
            "coherent": complex_cosine(coherent_state, outer_state),
            "phase_shuffled": complex_cosine(
                shuffled_prediction_state, outer_state
            ),
        }
        passed_fold = bool(
            ap_lift >= MIN_AP_LIFT
            and mass_lift >= MIN_MASS_COSINE_LIFT
            and control_ap > 0.0
            and control_mass > 0.0
            and renderer_delta["weighted"] >= MIN_RENDERER_DELTA
            and renderer_control["weighted"] > 0.0
        )
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "source_discovery": source_report,
                "fit_assignment": fit.report,
                "outer_assignment": outer.report,
                "beta": beta,
                "beta_report": beta_report,
                "fit_complex_state": fit_state_report,
                "outer_complex_state": outer_state_report,
                "coherent_prediction": coherent_report,
                "shuffled_prediction": shuffled_report,
                "complex_direction_metrics": complex_metric,
                "source_metrics": source_metrics,
                "source_lift_vs_mass_knn": {
                    "pair_ap": float(ap_lift),
                    "mass_cosine": float(mass_lift),
                },
                "source_lift_vs_phase_shuffle": {
                    "pair_ap": float(control_ap),
                    "mass_cosine": float(control_mass),
                },
                "renderer_metrics": renderer_metrics,
                "renderer_delta_vs_mass_knn": renderer_delta,
                "renderer_delta_vs_phase_shuffle": renderer_control,
                "passed_fold": passed_fold,
                "diagnostics": {
                    "mass_knn": {
                        key: summarize(value) for key, value in knn_diagnostics.items()
                    },
                    "coherent": {
                        key: summarize(value)
                        for key, value in coherent_diagnostics.items()
                    },
                    "phase_shuffled": {
                        key: summarize(value)
                        for key, value in shuffled_diagnostics.items()
                    },
                },
                "leakage_assertions": {
                    "outer_in_source_discovery": 0,
                    "outer_in_beta_fit": 0,
                    "outer_in_knn_or_complex_bank": 0,
                    "outer_complex_state_used_for_scoring_only": 1,
                },
            }
        )
        oof_payload[f"fold_{fold}_ids"] = outer_ids
        oof_payload[f"fold_{fold}_anchor"] = anchor
        oof_payload[f"fold_{fold}_mass_knn"] = knn_prediction
        oof_payload[f"fold_{fold}_coherent"] = coherent_channel
        oof_payload[f"fold_{fold}_phase_shuffled"] = shuffled_channel
        print(
            f"v363 fold={fold} beta={beta:.2f} AP={ap_lift:+.4f} "
            f"mass={mass_lift:+.4f} renderer={renderer_delta['weighted']:+.6f} "
            f"control={renderer_control['weighted']:+.6f} pass={passed_fold}",
            flush=True,
        )

    mean_source_lift = {
        key: float(
            np.mean([row["source_lift_vs_mass_knn"][key] for row in fold_rows])
        )
        for key in ("pair_ap", "mass_cosine")
    }
    mean_control_lift = {
        key: float(
            np.mean([row["source_lift_vs_phase_shuffle"][key] for row in fold_rows])
        )
        for key in ("pair_ap", "mass_cosine")
    }
    mean_renderer_delta = {
        key: float(
            np.mean([row["renderer_delta_vs_mass_knn"][key] for row in fold_rows])
        )
        for key in ("sh", "d", "weighted")
    }
    mean_renderer_control = {
        key: float(
            np.mean([row["renderer_delta_vs_phase_shuffle"][key] for row in fold_rows])
        )
        for key in ("sh", "d", "weighted")
    }
    fold_wins = int(sum(row["passed_fold"] for row in fold_rows))
    passed = bool(
        mean_source_lift["pair_ap"] >= MIN_AP_LIFT
        and mean_source_lift["mass_cosine"] >= MIN_MASS_COSINE_LIFT
        and mean_control_lift["pair_ap"] > 0.0
        and mean_control_lift["mass_cosine"] > 0.0
        and mean_renderer_delta["weighted"] >= MIN_RENDERER_DELTA
        and mean_renderer_control["weighted"] > 0.0
        and fold_wins >= MIN_FOLD_WINS
    )
    np.savez_compressed(OOF_OUT, **oof_payload)
    report = {
        "method": VERSION,
        "status": "strict_coherent_source_passed" if passed else "analysis_only_no_candidate",
        "round": 5,
        "primary_hypothesis": (
            "A true virtual source has a coherent official-H complex gain after fit-only carrier "
            "dephasing; coherent KNN should reinforce persistent paths and cancel false source mass "
            "that nonnegative mass interpolation cannot distinguish."
        ),
        "compliance": {
            "uses": [
                "official outer-fit H continuous-component center gains",
                "official train coordinates and hard fit-only source identities",
                "official Round1_Map point-cloud-derived source observations",
            ],
            "does_not_use": [
                "outer H in source discovery, beta fit, KNN, or complex prediction bank",
                "v207/v212/synthetic truth or teacher",
                "restored RT channel/material coefficients",
                "local or online labels",
            ],
        },
        "parameters": {
            "beta_grid": [float(v325.BETA_GRID[0]), float(v325.BETA_GRID[-1]), 0.02],
            "same_source_pair_neighbors": PAIR_NEIGHBORS,
            "maximum_pair_distance_m": MAX_PAIR_DISTANCE,
            "knn": v354.K_ANCHOR,
            "total_source_mass": "exactly preserved from v354 mass KNN",
            "phase_control": "independent random common phase per fit UE-source state",
            "parameter_search": "none beyond fit-only physical beta estimation",
            "minimum_pair_ap_lift": MIN_AP_LIFT,
            "minimum_mass_cosine_lift": MIN_MASS_COSINE_LIFT,
            "minimum_renderer_delta": MIN_RENDERER_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
        },
        "observation": observation_report,
        "fold_rows": fold_rows,
        "decision": {
            "mean_source_lift_vs_mass_knn": mean_source_lift,
            "mean_source_lift_vs_phase_shuffle": mean_control_lift,
            "mean_renderer_delta_vs_mass_knn": mean_renderer_delta,
            "mean_renderer_delta_vs_phase_shuffle": mean_renderer_control,
            "fold_wins": fold_wins,
            "passed": passed,
            "next_action": (
                "integrate coherent source state into the strict final-current chain"
                if passed
                else "stop after five rounds and reassess the source-lifecycle formulation with the user"
            ),
        },
        "outputs": {
            "oof": PATHS.relative(OOF_OUT),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
