#!/usr/bin/env python3
"""Strict fit-only fingerprint subfamily identifiability experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v311_effective_source_existence_boundary as v311
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v354_nested_virtual_source_transport as v354
import analyze_v355_virtual_source_lifecycle_model as v355
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 359
VERSION = "v359_fingerprint_subfamily_identifiability"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v359_fingerprint_subfamily_oof.npz"

MAX_SUBFAMILIES = 4
MIN_SUBFAMILY_UE = 5
MIN_PARENT_OBSERVATIONS_TO_SPLIT = 12
MIN_SILHOUETTE = 0.15
MIN_AP_LIFT = 0.05
MIN_MASS_COSINE_LIFT = 0.03
MIN_FOLD_WINS = 4


@dataclass
class AssignedObservations:
    rows: np.ndarray
    point_rows: np.ndarray
    parent: np.ndarray
    value: np.ndarray
    delay_residual: np.ndarray
    h_residual: np.ndarray
    target: np.ndarray
    report: dict[str, float]


@dataclass
class FamilyDefinition:
    parent: np.ndarray
    centers: np.ndarray
    prototype: np.ndarray
    fit_family: np.ndarray
    fit_family_shuffled: np.ndarray
    outer_family: np.ndarray
    outer_target: np.ndarray
    fit_target: np.ndarray
    shuffled_fit_target: np.ndarray
    report: dict[str, object]


def signed_circular_delta(value: np.ndarray, center: np.ndarray, period: float) -> np.ndarray:
    return np.remainder(value - center + period / 2.0, period) - period / 2.0


def assign_observations(
    observation: v311.Observations,
    query_ids: np.ndarray,
    centers: np.ndarray,
    train_pos: np.ndarray,
) -> AssignedObservations:
    """Match each query component to a parent source and retain assignment identity."""
    target = np.zeros((len(query_ids), len(centers)), dtype=np.float32)
    row_lookup = {int(point_id): row for row, point_id in enumerate(query_ids)}
    selected = np.isin(observation.point_id, query_ids)
    by_point: dict[int, list[int]] = {}
    for row in np.flatnonzero(selected):
        by_point.setdefault(int(observation.point_id[row]), []).append(int(row))

    source_first_length = np.linalg.norm(centers - v304.BS[:2], axis=1)
    source_h = v304.expected_h_bin(
        np.arctan2(centers[:, 1] - v304.BS[1], centers[:, 0] - v304.BS[0])
    )
    matched_rows: list[np.ndarray] = []
    matched_point_rows: list[np.ndarray] = []
    matched_parent: list[np.ndarray] = []
    matched_value: list[np.ndarray] = []
    matched_delay_residual: list[np.ndarray] = []
    matched_h_residual: list[np.ndarray] = []
    represented_mass = 0.0
    total_mass = float(np.sum(observation.mass[selected], dtype=np.float64))

    for point_id, rows in by_point.items():
        rows_array = np.asarray(rows, dtype=np.int64)
        predicted_length = source_first_length + np.linalg.norm(
            centers - train_pos[point_id, :2], axis=1
        )
        signed_delay = (
            observation.length[rows_array, None] - predicted_length[None, :]
        ) / float(v304.METERS_PER_BIN)
        h_delta = signed_circular_delta(
            observation.h[rows_array, None], source_h[None, :], 16.0
        )
        cost = (signed_delay / v311.MIN_LENGTH_ERROR) ** 2 + (
            h_delta / v311.MIN_H_ERROR
        ) ** 2
        observation_index, source_index = linear_sum_assignment(cost)
        keep = (
            (np.abs(signed_delay[observation_index, source_index]) <= v311.MIN_LENGTH_ERROR)
            & (np.abs(h_delta[observation_index, source_index]) <= v311.MIN_H_ERROR)
        )
        if not np.any(keep):
            continue
        observation_index = observation_index[keep]
        source_index = source_index[keep]
        selected_rows = rows_array[observation_index]
        confidence = np.exp(-0.5 * cost[observation_index, source_index]).astype(np.float32)
        value = observation.mass[selected_rows] * confidence
        point_row = np.full(len(selected_rows), row_lookup[point_id], dtype=np.int64)
        np.add.at(target, (point_row, source_index), value)

        matched_rows.append(selected_rows)
        matched_point_rows.append(point_row)
        matched_parent.append(source_index.astype(np.int64))
        matched_value.append(value.astype(np.float32))
        matched_delay_residual.append(
            signed_delay[observation_index, source_index].astype(np.float32)
        )
        matched_h_residual.append(h_delta[observation_index, source_index].astype(np.float32))
        represented_mass += float(np.sum(observation.mass[selected_rows], dtype=np.float64))

    def concatenate(values: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
        return np.concatenate(values).astype(dtype, copy=False) if values else np.empty(0, dtype=dtype)

    return AssignedObservations(
        rows=concatenate(matched_rows, np.int64),
        point_rows=concatenate(matched_point_rows, np.int64),
        parent=concatenate(matched_parent, np.int64),
        value=concatenate(matched_value, np.float32),
        delay_residual=concatenate(matched_delay_residual, np.float32),
        h_residual=concatenate(matched_h_residual, np.float32),
        target=target,
        report={
            "observation_count": int(np.sum(selected)),
            "matched_count": int(sum(len(value) for value in matched_rows)),
            "represented_unmatched_mass": represented_mass / max(total_mass, 1e-30),
        },
    )


def fingerprint_features(
    assigned: AssignedObservations,
    observation: v311.Observations,
    fingerprint: dict[str, np.ndarray],
) -> np.ndarray:
    point_id = observation.point_id[assigned.rows]
    component_id = observation.component_id[assigned.rows]

    def block(name: str, weight: float) -> np.ndarray:
        value = np.maximum(fingerprint[name][point_id, component_id], 0.0)
        value /= np.maximum(value.sum(axis=1, keepdims=True), 1e-20)
        return np.sqrt(value) * np.sqrt(weight)

    delay = np.clip(
        assigned.delay_residual / max(v311.MIN_LENGTH_ERROR, 1e-6), -1.0, 1.0
    )[:, None]
    horizontal = np.clip(
        assigned.h_residual / max(v311.MIN_H_ERROR, 1e-6), -1.0, 1.0
    )[:, None]
    return np.concatenate(
        [
            block("horizontal", 0.30),
            block("vertical", 0.30),
            block("polarization", 0.15),
            block("ue_port", 0.15),
            delay * np.sqrt(0.05),
            horizontal * np.sqrt(0.05),
        ],
        axis=1,
    ).astype(np.float32)


def choose_subfamilies(
    feature: np.ndarray,
    point_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Choose a conservative fit-only spherical partition for one parent source."""
    count = len(feature)
    base_label = np.zeros(count, dtype=np.int64)
    base_center = np.mean(feature, axis=0, keepdims=True).astype(np.float32)
    base = {
        "k": 1,
        "silhouette": 0.0,
        "counts": [count],
        "unique_ue": [int(len(np.unique(point_ids)))],
    }
    if count < MIN_PARENT_OBSERVATIONS_TO_SPLIT:
        return base_label, base_center, base

    best: tuple[float, np.ndarray, np.ndarray, dict[str, object]] | None = None
    max_k = min(MAX_SUBFAMILIES, count // MIN_SUBFAMILY_UE)
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=seed + 101 * k)
        label = model.fit_predict(feature)
        counts = [int(np.sum(label == family)) for family in range(k)]
        unique_ue = [
            int(len(np.unique(point_ids[label == family]))) for family in range(k)
        ]
        if min(unique_ue) < MIN_SUBFAMILY_UE:
            continue
        score = float(silhouette_score(feature, label, metric="euclidean"))
        report = {
            "k": k,
            "silhouette": score,
            "counts": counts,
            "unique_ue": unique_ue,
        }
        if best is None or score > best[0]:
            best = (score, label.astype(np.int64), model.cluster_centers_.astype(np.float32), report)
    if best is None or best[0] < MIN_SILHOUETTE:
        return base_label, base_center, base
    return best[1], best[2], best[3]


def define_families(
    fit: AssignedObservations,
    outer: AssignedObservations,
    observation: v311.Observations,
    centers: np.ndarray,
    fingerprint: dict[str, np.ndarray],
    seed: int,
) -> FamilyDefinition:
    fit_feature = fingerprint_features(fit, observation, fingerprint)
    outer_feature = fingerprint_features(outer, observation, fingerprint)
    fit_family = np.full(len(fit.rows), -1, dtype=np.int64)
    fit_family_shuffled = np.full(len(fit.rows), -1, dtype=np.int64)
    outer_family = np.full(len(outer.rows), -1, dtype=np.int64)
    family_parent: list[int] = []
    family_centers: list[np.ndarray] = []
    family_prototype: list[np.ndarray] = []
    parent_reports: list[dict[str, object]] = []
    offset = 0
    rng = np.random.default_rng(seed)

    for parent in range(len(centers)):
        fit_rows = np.flatnonzero(fit.parent == parent)
        outer_rows = np.flatnonzero(outer.parent == parent)
        point_ids = observation.point_id[fit.rows[fit_rows]]
        label, prototype, partition = choose_subfamilies(
            fit_feature[fit_rows], point_ids, seed + parent * 1009
        )
        count = len(prototype)
        global_label = label + offset
        fit_family[fit_rows] = global_label
        fit_family_shuffled[fit_rows] = label[rng.permutation(len(label))] + offset
        if len(outer_rows):
            distance = np.sum(
                (outer_feature[outer_rows, None, :] - prototype[None, :, :]) ** 2,
                axis=2,
            )
            outer_family[outer_rows] = np.argmin(distance, axis=1).astype(np.int64) + offset
        family_parent.extend([parent] * count)
        family_centers.extend([centers[parent]] * count)
        family_prototype.extend(prototype)
        parent_reports.append({"parent": parent, **partition})
        offset += count

    if np.any(fit_family < 0) or np.any(outer_family < 0):
        raise AssertionError("matched observation missing fingerprint family assignment")
    family_count = offset
    fit_target = np.zeros((fit.target.shape[0], family_count), dtype=np.float32)
    shuffled_fit_target = np.zeros_like(fit_target)
    outer_target = np.zeros((outer.target.shape[0], family_count), dtype=np.float32)
    np.add.at(fit_target, (fit.point_rows, fit_family), fit.value)
    np.add.at(shuffled_fit_target, (fit.point_rows, fit_family_shuffled), fit.value)
    np.add.at(outer_target, (outer.point_rows, outer_family), outer.value)
    parent_index = np.asarray(family_parent, dtype=np.int64)

    if not np.allclose(
        aggregate_parent(fit_target, parent_index, len(centers)), fit.target, atol=1e-6
    ):
        raise AssertionError("real family masses do not conserve parent target")
    if not np.allclose(
        aggregate_parent(shuffled_fit_target, parent_index, len(centers)), fit.target, atol=1e-6
    ):
        raise AssertionError("shuffled family masses do not conserve parent target")
    if not np.allclose(
        aggregate_parent(outer_target, parent_index, len(centers)), outer.target, atol=1e-6
    ):
        raise AssertionError("outer family masses do not conserve parent target")

    split = [row for row in parent_reports if int(row["k"]) > 1]
    return FamilyDefinition(
        parent=parent_index,
        centers=np.asarray(family_centers, dtype=np.float32),
        prototype=np.asarray(family_prototype, dtype=np.float32),
        fit_family=fit_family,
        fit_family_shuffled=fit_family_shuffled,
        outer_family=outer_family,
        outer_target=outer_target,
        fit_target=fit_target,
        shuffled_fit_target=shuffled_fit_target,
        report={
            "parent_count": int(len(centers)),
            "family_count": int(family_count),
            "split_parent_count": int(len(split)),
            "split_parent_fraction": float(len(split) / max(len(centers), 1)),
            "mean_selected_k": float(np.mean([row["k"] for row in parent_reports])),
            "split_silhouette": summarize(
                np.asarray([row["silhouette"] for row in split], dtype=np.float32)
            ),
            "parent_partitions": parent_reports,
        },
    )


def aggregate_parent(
    family_value: np.ndarray, family_parent: np.ndarray, parent_count: int
) -> np.ndarray:
    output = np.zeros((len(family_value), parent_count), dtype=np.float32)
    for family, parent in enumerate(family_parent):
        output[:, parent] += family_value[:, family]
    return output


def identity_pair_features(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    bank_target: np.ndarray,
    centers: np.ndarray,
    identity: np.ndarray,
    train_pos: np.ndarray,
    scene_feature: np.ndarray,
    signature: np.ndarray,
    scene: dict[str, np.ndarray],
) -> np.ndarray:
    physical = v355.pair_features(
        query_ids,
        bank_ids,
        bank_target,
        centers,
        train_pos,
        scene_feature,
        signature,
        scene,
    )
    repeated = np.broadcast_to(identity[None, :, :], physical.shape[:2] + (identity.shape[1],))
    return np.concatenate([physical, repeated], axis=2).astype(np.float32)


def crossfit_family_features(
    fit_ids: np.ndarray,
    fit_target: np.ndarray,
    centers: np.ndarray,
    identity: np.ndarray,
    train_pos: np.ndarray,
    scene_feature: np.ndarray,
    signature: np.ndarray,
    scene: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    row_lookup = np.full(len(train_pos), -1, dtype=np.int64)
    row_lookup[fit_ids] = np.arange(len(fit_ids), dtype=np.int64)
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for query_ids in v315.disjoint_folds(fit_ids, train_pos[:, :2]):
        bank_ids = np.setdiff1d(fit_ids, query_ids)
        bank_rows = row_lookup[bank_ids]
        query_rows = row_lookup[query_ids]
        feature_parts.append(
            identity_pair_features(
                query_ids,
                bank_ids,
                fit_target[bank_rows],
                centers,
                identity,
                train_pos,
                scene_feature,
                signature,
                scene,
            )
        )
        target_parts.append(fit_target[query_rows])
    return np.concatenate(feature_parts), np.concatenate(target_parts)


def fit_and_predict(
    fit_ids: np.ndarray,
    outer_ids: np.ndarray,
    fit_target: np.ndarray,
    centers: np.ndarray,
    identity: np.ndarray,
    train_pos: np.ndarray,
    scene_feature: np.ndarray,
    signature: np.ndarray,
    scene: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    train_feature, train_target = crossfit_family_features(
        fit_ids,
        fit_target,
        centers,
        identity,
        train_pos,
        scene_feature,
        signature,
        scene,
    )
    classifier, regressor, training = v355.fit_models(train_feature, train_target, seed)
    outer_feature = identity_pair_features(
        outer_ids,
        fit_ids,
        fit_target,
        centers,
        identity,
        train_pos,
        scene_feature,
        signature,
        scene,
    )
    prediction, probability, conditional_mass = v355.predict_mass(
        classifier, regressor, outer_feature
    )
    return prediction, {
        "training": training,
        "probability": summarize(probability),
        "conditional_mass": summarize(conditional_mass),
    }


def continuity_metrics(
    fit: AssignedObservations,
    outer: AssignedObservations,
    families: FamilyDefinition,
    observation: v311.Observations,
    train_pos: np.ndarray,
) -> dict[str, float]:
    """Measure family identity continuity conditioned on true parent presence."""
    correct_real = 0.0
    correct_shuffle = 0.0
    total = 0.0
    for outer_row in range(len(outer.rows)):
        parent = int(outer.parent[outer_row])
        candidate = np.flatnonzero(fit.parent == parent)
        if not len(candidate):
            continue
        outer_point = observation.point_id[outer.rows[outer_row]]
        fit_points = observation.point_id[fit.rows[candidate]]
        distance = np.linalg.norm(
            train_pos[fit_points, :2] - train_pos[outer_point, :2], axis=1
        )
        nearest = candidate[int(np.argmin(distance))]
        weight = float(outer.value[outer_row])
        correct_real += weight * float(
            families.fit_family[nearest] == families.outer_family[outer_row]
        )
        correct_shuffle += weight * float(
            families.fit_family_shuffled[nearest] == families.outer_family[outer_row]
        )
        total += weight
    return {
        "nearest_support_mass_accuracy": correct_real / max(total, 1e-30),
        "shuffled_nearest_support_mass_accuracy": correct_shuffle / max(total, 1e-30),
        "lift_over_shuffle": (correct_real - correct_shuffle) / max(total, 1e-30),
    }


def main() -> None:
    for path in (
        v354.TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v300.FINGERPRINT_CACHE,
        v354.TRAIN_CANDIDATES,
        v273.SCENE_CACHE,
        v273.FEATURE_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    target = np.load(v354.TARGET_CACHE)
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    observation, observation_report = v354.build_physical_observations()
    fingerprint_data = np.load(v300.FINGERPRINT_CACHE)
    fingerprint = {key: fingerprint_data[key] for key in fingerprint_data.files}
    scene_data = np.load(v273.SCENE_CACHE)
    scene = {key: scene_data[key] for key in scene_data.files}
    scene_feature = scene["train_features"].astype(np.float32)
    signature = np.load(v273.FEATURE_CACHE)["train_signature"].astype(np.float32)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])

    fold_rows: list[dict[str, object]] = []
    oof_ids: list[np.ndarray] = []
    oof_parent_target: list[np.ndarray] = []
    oof_knn: list[np.ndarray] = []
    oof_real: list[np.ndarray] = []
    oof_shuffle: list[np.ndarray] = []

    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        centers, source_report = v311.discover_sources(observation, fit_ids)
        fit = assign_observations(observation, fit_ids, centers, train_pos)
        outer = assign_observations(observation, outer_ids, centers, train_pos)
        reference_fit, _ = v311.assign_targets(observation, fit_ids, centers, train_pos)
        reference_outer, _ = v311.assign_targets(observation, outer_ids, centers, train_pos)
        if not np.allclose(fit.target, reference_fit, atol=1e-6) or not np.allclose(
            outer.target, reference_outer, atol=1e-6
        ):
            raise AssertionError("v359 assignment does not reproduce v354 parent targets")

        family = define_families(
            fit,
            outer,
            observation,
            centers,
            fingerprint,
            VERSION_NUMBER * 10_000 + fold,
        )
        real_family_prediction, real_diagnostics = fit_and_predict(
            fit_ids,
            outer_ids,
            family.fit_target,
            family.centers,
            family.prototype,
            train_pos,
            scene_feature,
            signature,
            scene,
            VERSION_NUMBER + fold,
        )
        shuffled_family_prediction, shuffled_diagnostics = fit_and_predict(
            fit_ids,
            outer_ids,
            family.shuffled_fit_target,
            family.centers,
            family.prototype,
            train_pos,
            scene_feature,
            signature,
            scene,
            VERSION_NUMBER + 100 + fold,
        )
        parent_real = aggregate_parent(real_family_prediction, family.parent, len(centers))
        parent_shuffle = aggregate_parent(
            shuffled_family_prediction, family.parent, len(centers)
        )
        parent_knn = v311.knn_prediction(
            outer_ids, fit_ids, fit.target, train_pos, k=v354.K_ANCHOR
        )
        metrics = {
            "knn": v311.evaluate_prediction(parent_knn, outer.target),
            "fingerprint_subfamily": v311.evaluate_prediction(parent_real, outer.target),
            "shuffled_subfamily": v311.evaluate_prediction(parent_shuffle, outer.target),
        }
        ap_lift = (
            metrics["fingerprint_subfamily"]["pair_ap"] - metrics["knn"]["pair_ap"]
        )
        mass_lift = (
            metrics["fingerprint_subfamily"]["mass_cosine"]
            - metrics["knn"]["mass_cosine"]
        )
        control_ap_lift = (
            metrics["fingerprint_subfamily"]["pair_ap"]
            - metrics["shuffled_subfamily"]["pair_ap"]
        )
        control_mass_lift = (
            metrics["fingerprint_subfamily"]["mass_cosine"]
            - metrics["shuffled_subfamily"]["mass_cosine"]
        )
        passed_fold = bool(
            ap_lift >= MIN_AP_LIFT
            and mass_lift >= MIN_MASS_COSINE_LIFT
            and control_ap_lift > 0.0
            and control_mass_lift > 0.0
        )
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "source_discovery": source_report,
                "fit_assignment": fit.report,
                "outer_assignment": outer.report,
                "family_definition": family.report,
                "continuity": continuity_metrics(
                    fit, outer, family, observation, train_pos
                ),
                "metrics": metrics,
                "lift_vs_knn": {
                    "pair_ap": float(ap_lift),
                    "mass_cosine": float(mass_lift),
                },
                "lift_vs_shuffle": {
                    "pair_ap": float(control_ap_lift),
                    "mass_cosine": float(control_mass_lift),
                },
                "passed_fold": passed_fold,
                "real_model": real_diagnostics,
                "shuffle_model": shuffled_diagnostics,
                "leakage_assertions": {
                    "outer_in_source_discovery": 0,
                    "outer_fingerprint_used_as_prediction_input": 0,
                    "outer_fingerprint_used_for_scoring_label_only": 1,
                    "parent_mass_conservation_max_error": float(
                        np.max(
                            np.abs(
                                aggregate_parent(
                                    family.outer_target,
                                    family.parent,
                                    len(centers),
                                )
                                - outer.target
                            )
                        )
                    ),
                },
            }
        )
        oof_ids.append(outer_ids)
        oof_parent_target.append(outer.target)
        oof_knn.append(parent_knn)
        oof_real.append(parent_real)
        oof_shuffle.append(parent_shuffle)
        print(
            f"v359 fold={fold} parent={len(centers)} family={len(family.parent)} "
            f"AP={ap_lift:+.4f} mass={mass_lift:+.4f} "
            f"control_AP={control_ap_lift:+.4f} pass={passed_fold}",
            flush=True,
        )

    fold_wins = int(sum(row["passed_fold"] for row in fold_rows))
    mean_lift = {
        key: float(np.mean([row["lift_vs_knn"][key] for row in fold_rows]))
        for key in ("pair_ap", "mass_cosine")
    }
    mean_control = {
        key: float(np.mean([row["lift_vs_shuffle"][key] for row in fold_rows]))
        for key in ("pair_ap", "mass_cosine")
    }
    passed = bool(
        mean_lift["pair_ap"] >= MIN_AP_LIFT
        and mean_lift["mass_cosine"] >= MIN_MASS_COSINE_LIFT
        and fold_wins >= MIN_FOLD_WINS
        and mean_control["pair_ap"] > 0.0
        and mean_control["mass_cosine"] > 0.0
    )
    oof_payload: dict[str, np.ndarray] = {}
    for fold in range(len(folds)):
        oof_payload[f"fold_{fold}_ids"] = oof_ids[fold]
        oof_payload[f"fold_{fold}_parent_target"] = oof_parent_target[fold]
        oof_payload[f"fold_{fold}_knn"] = oof_knn[fold]
        oof_payload[f"fold_{fold}_fingerprint_subfamily"] = oof_real[fold]
        oof_payload[f"fold_{fold}_shuffled_subfamily"] = oof_shuffle[fold]
    np.savez_compressed(OOF_OUT, **oof_payload)
    report = {
        "method": VERSION,
        "status": "strict_identifiability_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A 2-D virtual-source cluster mixes propagation subfamilies whose stable official-H "
            "array/polarization fingerprints define route-specific visibility states predictable "
            "from coordinates, point-cloud route state, and fit-only support boundaries."
        ),
        "compliance": {
            "uses": [
                "official train H continuous components and path fingerprints",
                "official train coordinates",
                "official Round1_Map point-cloud scene and route features",
            ],
            "does_not_use": [
                "outer H in source discovery, fit support, model training, or prediction input",
                "v207/v212/synthetic truth or teacher",
                "restored scene RT channel/material coefficients",
                "local or online labels",
            ],
        },
        "parameters": {
            "max_subfamilies": MAX_SUBFAMILIES,
            "minimum_subfamily_unique_ue": MIN_SUBFAMILY_UE,
            "minimum_parent_observations_to_split": MIN_PARENT_OBSERVATIONS_TO_SPLIT,
            "minimum_silhouette": MIN_SILHOUETTE,
            "predictor": "frozen v355 cross-fit HistGradientBoosting lifecycle model",
            "fingerprint_weights": {
                "horizontal": 0.30,
                "vertical": 0.30,
                "polarization": 0.15,
                "ue_port": 0.15,
                "delay_residual": 0.05,
                "horizontal_residual": 0.05,
            },
            "parameter_search": "none",
        },
        "observation": observation_report,
        "fold_rows": fold_rows,
        "decision": {
            "minimum_pair_ap_lift": MIN_AP_LIFT,
            "minimum_mass_cosine_lift": MIN_MASS_COSINE_LIFT,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "mean_lift_vs_knn": mean_lift,
            "mean_lift_vs_shuffle": mean_control,
            "fold_wins": fold_wins,
            "passed": passed,
            "next_action": (
                "connect the learned subfamily selector to the frozen v354 renderer"
                if passed
                else "reject fingerprint subfamily selector before renderer integration"
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
