#!/usr/bin/env python3
"""Strict fit-only virtual-source subtract/transport/add experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v308_joint_matched_patch_renderer as v308
import analyze_v311_effective_source_existence_boundary as v311
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v274_planar_barrier_attention as v274
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 354
VERSION = "v354_nested_virtual_source_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v354_nested_virtual_source_oof.npz"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
TRAIN_CANDIDATES = v305.TRAIN_CANDIDATES
K_ANCHOR = 16
MIN_COMPONENT_MASS = 2e-3
MIN_MEAN_DELTA = 0.01
MIN_FOLD_WINS = 4
S_SIZE = 192
H_SIZE = 16
BATCH_SIZE = 48


@dataclass
class NeighborSet:
    ids: np.ndarray
    weights: np.ndarray


def normalize(x: np.ndarray) -> np.ndarray:
    return v297.normalize_rows(np.maximum(np.asarray(x, dtype=np.float32), 0.0))


def derive_d(sh: np.ndarray) -> np.ndarray:
    s = np.sum(np.asarray(sh, dtype=np.float32), axis=2, dtype=np.float64)
    return normalize(s[:, (-np.arange(S_SIZE)) % S_SIZE, None])[:, :, 0]


def metrics(sh: np.ndarray, truth_sh: np.ndarray, truth_d: np.ndarray) -> dict[str, float]:
    d = derive_d(sh)
    sh_score = float(np.mean(v297.row_cos(sh, truth_sh)))
    d_score = float(np.mean(v297.row_cos(d, truth_d)))
    return {
        "sh": sh_score,
        "d": d_score,
        "weighted": 0.52 * sh_score + 0.48 * d_score,
    }


def exact_atoms(center: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(center, dtype=np.float64)
    bins = np.arange(size, dtype=np.float64)
    delta = bins.reshape((1,) * value.ndim + (size,)) - value[..., None]
    delta = np.remainder(delta + size / 2.0, size) - size / 2.0
    atom = np.square(np.sinc(delta) / np.sinc(delta / size))
    atom /= np.maximum(atom.sum(axis=-1, keepdims=True), 1e-30)
    return atom.astype(np.float32)


def build_physical_observations() -> tuple[v311.Observations, dict[str, object]]:
    component = np.load(v297.COMPONENT_CACHE)
    fingerprint = np.load(v300.FINGERPRINT_CACHE)
    candidates_data = np.load(TRAIN_CANDIDATES)
    candidates = {key: candidates_data[key] for key in candidates_data.files}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    delay = component["delay"].astype(np.float32)
    mass = component["weight"].astype(np.float32)
    valid = component["valid"].astype(bool)
    horizontal = fingerprint["horizontal"].astype(np.float32)
    h_center = v304.horizontal_center(horizontal)
    los_component, _confidence, _error = v304.select_los_components(
        train_pos, delay, mass, valid, horizontal
    )
    physical_length = delay * float(v304.METERS_PER_BIN)
    match_observation = {
        "delay": delay,
        "mass": mass,
        "valid": valid,
        "h_center": h_center,
        "los_component": los_component,
        "path_length": physical_length,
    }
    matches = v305.collect_matches(match_observation, candidates)
    one_bounce = np.zeros(valid.shape, dtype=bool)
    good = (
        (matches["length_error_bins"] <= 0.50)
        & (matches["h_error_bins"] <= 1.00)
    )
    one_bounce[
        matches["point_id"][good], matches["component_id"][good].astype(np.int64)
    ] = True

    patches = np.load(v274.SCENE_CACHE)["wall_patches"].astype(np.float64)
    points: list[np.ndarray] = []
    point_ids: list[int] = []
    component_ids: list[int] = []
    masses: list[float] = []
    lengths: list[float] = []
    h_values: list[float] = []
    for point_id in range(len(train_pos)):
        direct = float(np.linalg.norm(train_pos[point_id, :2] - v304.BS[:2]))
        active = np.flatnonzero(
            valid[point_id]
            & (mass[point_id] >= MIN_COMPONENT_MASS)
            & ~one_bounce[point_id]
        )
        active = active[active != los_component[point_id]]
        for component_id in active:
            total_length = float(physical_length[point_id, component_id])
            if total_length <= direct + 1.0 or total_length > 2500.0:
                continue
            candidates_xy = [
                v304.reflection_point(
                    train_pos[point_id, :2], total_length, azimuth
                )
                for azimuth in v304.h_bin_azimuth_solutions(
                    float(h_center[point_id, component_id])
                )
            ]
            candidates_xy = [value for value in candidates_xy if value is not None]
            if not candidates_xy:
                continue
            candidate_array = np.asarray(candidates_xy, dtype=np.float64)
            wall_distance, _patch = v304.wall_distance(candidate_array, patches)
            selected = int(np.argmin(wall_distance))
            points.append(candidate_array[selected].astype(np.float32))
            point_ids.append(point_id)
            component_ids.append(int(component_id))
            masses.append(float(mass[point_id, component_id]))
            lengths.append(total_length)
            h_values.append(float(h_center[point_id, component_id]))
    observation = v311.Observations(
        point=np.asarray(points, dtype=np.float32),
        point_id=np.asarray(point_ids, dtype=np.int64),
        component_id=np.asarray(component_ids, dtype=np.int64),
        mass=np.asarray(masses, dtype=np.float32),
        length=np.asarray(lengths, dtype=np.float32),
        h=np.asarray(h_values, dtype=np.float32),
    )
    return observation, {
        "count": int(len(points)),
        "unique_ue": int(len(np.unique(observation.point_id))),
        "mass": summarize(observation.mass),
        "one_bounce_match_count": int(np.sum(one_bounce)),
        "delay_conversion": "fixed c/(N*subcarrier_spacing); no fitted H calibration",
    }


def neighbors(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    train_xy: np.ndarray,
    exclude_self: bool,
) -> NeighborSet:
    take = min(K_ANCHOR + (1 if exclude_self else 0), len(bank_ids))
    distance, local = cKDTree(train_xy[bank_ids]).query(train_xy[query_ids], k=take)
    if take == 1:
        distance = distance[:, None]
        local = local[:, None]
    candidate = bank_ids[local]
    selected_count = min(K_ANCHOR, len(bank_ids) - (1 if exclude_self else 0))
    selected_ids = np.empty((len(query_ids), selected_count), dtype=np.int64)
    selected_distance = np.empty((len(query_ids), selected_count), dtype=np.float64)
    for row, point_id in enumerate(query_ids):
        keep = candidate[row] != point_id if exclude_self else np.ones(take, dtype=bool)
        chosen = candidate[row, keep][:selected_count]
        chosen_distance = distance[row, keep][:selected_count]
        if len(chosen) != selected_count:
            raise RuntimeError("insufficient non-self KNN donors")
        selected_ids[row] = chosen
        selected_distance[row] = chosen_distance
    weight = 1.0 / np.maximum(selected_distance, 0.05) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return NeighborSet(selected_ids, weight.astype(np.float32))


def anchor_from_neighbors(bank: np.ndarray, neighbor: NeighborSet) -> np.ndarray:
    return normalize(
        np.sum(
            neighbor.weights[:, :, None, None] * bank[neighbor.ids],
            axis=1,
            dtype=np.float64,
        )
    )


def source_geometry(
    ue_xy: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first = np.linalg.norm(centers - v304.BS[:2], axis=1)
    second = np.linalg.norm(ue_xy[:, None, :] - centers[None, :, :], axis=2)
    delay = (first[None, :] + second) / float(v304.METERS_PER_BIN)
    azimuth = np.arctan2(
        centers[:, 1] - v304.BS[1], centers[:, 0] - v304.BS[0]
    )
    h = v304.expected_h_bin(azimuth)
    return delay.astype(np.float32), h.astype(np.float32)


def render_query_sources(
    mass: np.ndarray, delay: np.ndarray, h: np.ndarray
) -> np.ndarray:
    s_atom = exact_atoms(np.remainder(-delay, S_SIZE), S_SIZE)
    h_atom = exact_atoms(h, H_SIZE)
    return np.einsum("qs,qsd,sh->qdh", mass, s_atom, h_atom, optimize=True).astype(
        np.float32
    )


def transport(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    neighbor: NeighborSet,
    anchor: np.ndarray,
    fit_target: np.ndarray,
    centers: np.ndarray,
    train_xy: np.ndarray,
    query_mass_override: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    fit_lookup = np.full(len(train_xy), -1, dtype=np.int64)
    fit_lookup[fit_ids] = np.arange(len(fit_ids), dtype=np.int64)
    donor_rows = fit_lookup[neighbor.ids]
    if np.any(donor_rows < 0):
        raise AssertionError("KNN donor outside outer-fit bank")
    query_delay, h = source_geometry(train_xy[query_ids], centers)
    predicted_mass = np.sum(
        neighbor.weights[:, :, None] * fit_target[donor_rows],
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    rendered_mass = (
        predicted_mass
        if query_mass_override is None
        else np.asarray(query_mass_override, dtype=np.float32)
    )
    query_source = render_query_sources(rendered_mass, query_delay, h)

    donor_source = np.zeros_like(anchor, dtype=np.float32)
    for start in range(0, len(query_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(query_ids))
        ids = neighbor.ids[start:end]
        rows = donor_rows[start:end]
        donor_delay, _ = source_geometry(train_xy[ids.reshape(-1)], centers)
        donor_delay = donor_delay.reshape(end - start, ids.shape[1], len(centers))
        s_atom = exact_atoms(np.remainder(-donor_delay, S_SIZE), S_SIZE)
        h_atom = exact_atoms(h, H_SIZE)
        donor_source[start:end] = np.einsum(
            "qk,qks,qksd,sh->qdh",
            neighbor.weights[start:end],
            fit_target[rows],
            s_atom,
            h_atom,
            optimize=True,
        ).astype(np.float32)
    prediction = normalize(anchor + query_source - donor_source)
    return prediction, {
        "predicted_mass": predicted_mass.sum(axis=1).astype(np.float32),
        "donor_mass": donor_source.sum(axis=(1, 2)).astype(np.float32),
        "query_mass": query_source.sum(axis=(1, 2)).astype(np.float32),
        "clipped_negative_mass": np.maximum(
            -(anchor + query_source - donor_source), 0.0
        ).sum(axis=(1, 2)).astype(np.float32),
    }


def main() -> None:
    for path in (
        TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v300.FINGERPRINT_CACHE,
        TRAIN_CANDIDATES,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    target = np.load(TARGET_CACHE)
    truth_sh = normalize(target["sh"].astype(np.float32))
    truth_d = normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    train_xy = train_pos[:, :2]
    observation, observation_report = build_physical_observations()
    folds = v315.disjoint_folds(valid_ids, train_xy)
    if len(np.unique(np.concatenate(folds))) != len(valid_ids):
        raise AssertionError("outer folds are not a disjoint cover of valid IDs")

    fold_rows: list[dict[str, object]] = []
    oof_ids: list[np.ndarray] = []
    oof_anchor: list[np.ndarray] = []
    oof_prediction: list[np.ndarray] = []
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        source_observation_ids = np.unique(
            observation.point_id[np.isin(observation.point_id, fit_ids)]
        )
        if np.intersect1d(source_observation_ids, outer_ids).size:
            raise AssertionError("outer H entered source discovery")
        centers, source_report = v311.discover_sources(observation, fit_ids)
        fit_target, fit_assignment = v311.assign_targets(
            observation, fit_ids, centers, train_pos
        )
        outer_target, outer_assignment = v311.assign_targets(
            observation, outer_ids, centers, train_pos
        )
        outer_neighbor = neighbors(outer_ids, fit_ids, train_xy, False)
        if np.intersect1d(np.unique(outer_neighbor.ids), outer_ids).size:
            raise AssertionError("outer H entered KNN current")
        anchor = anchor_from_neighbors(truth_sh, outer_neighbor)
        prediction, diagnostics = transport(
            outer_ids,
            fit_ids,
            outer_neighbor,
            anchor,
            fit_target,
            centers,
            train_xy,
        )
        oracle_prediction, oracle_diagnostics = transport(
            outer_ids,
            fit_ids,
            outer_neighbor,
            anchor,
            fit_target,
            centers,
            train_xy,
            query_mass_override=outer_target,
        )
        base_metric = metrics(anchor, truth_sh[outer_ids], truth_d[outer_ids])
        model_metric = metrics(
            prediction, truth_sh[outer_ids], truth_d[outer_ids]
        )
        delta = {
            key: float(model_metric[key] - base_metric[key])
            for key in ("sh", "d", "weighted")
        }
        oracle_metric = metrics(
            oracle_prediction, truth_sh[outer_ids], truth_d[outer_ids]
        )
        oracle_delta = {
            key: float(oracle_metric[key] - base_metric[key])
            for key in ("sh", "d", "weighted")
        }
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "source_discovery": source_report,
                "fit_assignment": fit_assignment,
                "outer_assignment": outer_assignment,
                "outer_source_mass": summarize(outer_target.sum(axis=1)),
                "baseline": base_metric,
                "prediction": model_metric,
                "delta": delta,
                "oracle_query_mass": oracle_metric,
                "oracle_query_mass_delta": oracle_delta,
                "diagnostics": {
                    key: summarize(value) for key, value in diagnostics.items()
                },
                "oracle_diagnostics": {
                    key: summarize(value)
                    for key, value in oracle_diagnostics.items()
                },
                "leakage_assertions": {
                    "outer_in_source_discovery": 0,
                    "outer_in_knn_bank": 0,
                    "source_discovery_ids": int(len(source_observation_ids)),
                    "knn_unique_donor_ids": int(len(np.unique(outer_neighbor.ids))),
                },
            }
        )
        oof_ids.append(outer_ids)
        oof_anchor.append(anchor)
        oof_prediction.append(prediction)
        print(
            f"v354 fold={fold} sources={len(centers)} "
            f"mass={outer_target.sum(axis=1).mean():.4f} "
            f"delta={delta['weighted']:+.6f} "
            f"oracle={oracle_delta['weighted']:+.6f}",
            flush=True,
        )

    mean_delta = {
        key: float(np.mean([row["delta"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    wins = int(sum(row["delta"]["weighted"] > 0.0 for row in fold_rows))
    oracle_mean_delta = {
        key: float(
            np.mean([row["oracle_query_mass_delta"][key] for row in fold_rows])
        )
        for key in ("sh", "d", "weighted")
    }
    passed = bool(mean_delta["weighted"] >= MIN_MEAN_DELTA and wins >= MIN_FOLD_WINS)
    np.savez_compressed(
        OOF_OUT,
        ids=np.concatenate(oof_ids),
        anchor=np.concatenate(oof_anchor),
        prediction=np.concatenate(oof_prediction),
    )
    report = {
        "method": VERSION,
        "status": "strict_oof_breakthrough" if passed else "analysis_only_no_candidate",
        "round": 1,
        "primary_hypothesis": (
            "Fit-only virtual sources identify propagation families whose neighbor component "
            "mass can be subtracted at donor delay/angle and transported to the query geometry."
        ),
        "compliance": {
            "uses": [
                "official train H continuous components",
                "official train coordinates",
                "official Round1_Map point-cloud wall candidates",
                "fixed physical delay-bin conversion",
            ],
            "does_not_use": [
                "outer-fold H in source discovery, labels, KNN current, or selection",
                "v207/v212/synthetic truth or teacher",
                "restored scene/material/RT coefficients",
                "local or online labels",
            ],
        },
        "parameters": {
            "fold_design": "five mutually exclusive isolation-stratified folds",
            "knn": K_ANCHOR,
            "cluster_eps_m": v311.CLUSTER_EPS,
            "cluster_min_samples": v311.CLUSTER_MIN_SAMPLES,
            "minimum_unique_ue": v311.MIN_UNIQUE_UE,
            "transport_alpha": 1.0,
            "parameter_search": "none",
            "minimum_mean_delta": MIN_MEAN_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
        },
        "observation": observation_report,
        "fold_rows": fold_rows,
        "decision": {
            "mean_delta": mean_delta,
            "fold_wins": wins,
            "passed": passed,
            "oracle_query_mass_mean_delta": oracle_mean_delta,
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
