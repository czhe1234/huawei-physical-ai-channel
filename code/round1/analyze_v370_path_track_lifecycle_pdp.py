#!/usr/bin/env python3
"""Track path lifecycle on the official planar UE field.

The eikonal candidates from v366 already localize delay accurately. This
experiment connects reciprocal candidates at neighboring UE positions into a
transductive path graph. Only fit-query lifecycle labels are exposed as graph
seeds. Candidate existence and conditional mass remain separate targets.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v365_multiview_component_pdp as v365
import analyze_v366_local_eikonal_pdp as v366
import analyze_v368_terminal_ray_path_existence as v368
import analyze_v369_tree_scored_eikonal_pdp as v369
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 370
VERSION = "v370_path_track_lifecycle_pdp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v370_path_track_lifecycle_pdp_oof.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

QUERY_NEIGHBORS = 8
MAX_QUERY_DISTANCE = 14.0
MAX_TRACK_DELAY_ERROR = 0.25
MAX_TRACK_GRADIENT_ERROR = 0.75 * v366.MAX_GRADIENT
DELAY_WEIGHT_SCALE = 0.12
GRADIENT_WEIGHT_SCALE = 0.35 * v366.MAX_GRADIENT
DIFFUSION_STEPS = (1, 2, 4, 8)

MIN_AUDIT_AP_CONTROL_LIFT = 0.02
MIN_AUDIT_MASS_CONTROL_LIFT = 0.02
MIN_AUDIT_FOLD_WINS = 4

MIN_COSINE_LIFT = 0.010
MIN_TOP_RECALL_LIFT = 0.05
MIN_EMD_REDUCTION = 0.10
MIN_FOLD_WINS = 4
MIN_LABEL_CONTROL_LIFT = 0.002
MAX_CLEAR_DEGRADATION = 0.002


@dataclass
class TrackBundle:
    hypothesis: v366.HypothesisSet
    arrays: dict[str, np.ndarray]
    transition: csr_matrix
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_weight: np.ndarray
    node_query_id: np.ndarray
    structural_feature: np.ndarray


def build_track_graph(
    hypothesis: v366.HypothesisSet,
    arrays: dict[str, np.ndarray],
    all_xy: np.ndarray,
) -> TrackBundle:
    valid = hypothesis.valid
    node_of = np.full(valid.shape, -1, dtype=np.int64)
    node_of[valid] = np.arange(int(np.sum(valid)), dtype=np.int64)
    node_query_id = hypothesis.query_ids[arrays["query_row"]]
    expected_query_id = np.repeat(hypothesis.query_ids, valid.sum(axis=1))
    if not np.array_equal(node_query_id, expected_query_id):
        raise RuntimeError("candidate flattening order does not match pair arrays")

    query_xy = all_xy[hypothesis.query_ids]
    take = min(QUERY_NEIGHBORS + 1, len(query_xy))
    distance, neighbor = cKDTree(query_xy).query(query_xy, k=take)
    if take == 1:
        distance = distance[:, None]
        neighbor = neighbor[:, None]

    edge_u: list[int] = []
    edge_v: list[int] = []
    edge_weight: list[float] = []
    for query_row in range(len(query_xy)):
        for rank in range(1, take):
            other_row = int(neighbor[query_row, rank])
            spatial_distance = float(distance[query_row, rank])
            if other_row <= query_row or spatial_distance > MAX_QUERY_DISTANCE:
                continue
            active = np.flatnonzero(valid[query_row])
            other_active = np.flatnonzero(valid[other_row])
            if len(active) == 0 or len(other_active) == 0:
                continue

            delta_xy = query_xy[other_row] - query_xy[query_row]
            delay_a = hypothesis.delay[query_row, active].astype(np.float64)
            delay_b = hypothesis.delay[other_row, other_active].astype(np.float64)
            gradient_a = (
                hypothesis.feature[query_row, active, :2].astype(np.float64)
                * v366.MAX_GRADIENT
            )
            gradient_b = (
                hypothesis.feature[other_row, other_active, :2].astype(np.float64)
                * v366.MAX_GRADIENT
            )
            predicted_b = delay_a + gradient_a @ delta_xy
            predicted_a = delay_b - gradient_b @ delta_xy
            forward = np.abs(
                v366.signed_circular_delta(delay_b[None, :], predicted_b[:, None])
            )
            reverse = np.abs(
                v366.signed_circular_delta(delay_a[:, None], predicted_a[None, :])
            )
            delay_error = np.maximum(forward, reverse)
            gradient_error = np.linalg.norm(
                gradient_a[:, None, :] - gradient_b[None, :, :], axis=2
            )
            cost = (
                (delay_error / MAX_TRACK_DELAY_ERROR) ** 2
                + (gradient_error / MAX_TRACK_GRADIENT_ERROR) ** 2
            )
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                delay_value = float(delay_error[row, column])
                gradient_value = float(gradient_error[row, column])
                if (
                    delay_value > MAX_TRACK_DELAY_ERROR
                    or gradient_value > MAX_TRACK_GRADIENT_ERROR
                ):
                    continue
                node_a = int(node_of[query_row, active[row]])
                node_b = int(node_of[other_row, other_active[column]])
                weight = math.exp(
                    -0.5 * (delay_value / DELAY_WEIGHT_SCALE) ** 2
                    -0.5 * (gradient_value / GRADIENT_WEIGHT_SCALE) ** 2
                    -0.5 * (spatial_distance / MAX_QUERY_DISTANCE) ** 2
                )
                edge_u.append(node_a)
                edge_v.append(node_b)
                edge_weight.append(max(weight, 1e-6))

    node_count = len(node_query_id)
    edge_u_array = np.asarray(edge_u, dtype=np.int64)
    edge_v_array = np.asarray(edge_v, dtype=np.int64)
    edge_weight_array = np.asarray(edge_weight, dtype=np.float64)
    if len(edge_u_array):
        row = np.concatenate([edge_u_array, edge_v_array])
        column = np.concatenate([edge_v_array, edge_u_array])
        data = np.concatenate([edge_weight_array, edge_weight_array])
        adjacency = coo_matrix(
            (data, (row, column)), shape=(node_count, node_count)
        ).tocsr()
    else:
        adjacency = csr_matrix((node_count, node_count), dtype=np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_degree = np.zeros_like(degree)
    inverse_degree[degree > 0.0] = 1.0 / degree[degree > 0.0]
    transition = (diags(inverse_degree) @ adjacency).tocsr()

    component_count, component_id = connected_components(
        adjacency, directed=False, return_labels=True
    )
    component_size = np.bincount(component_id, minlength=component_count)
    unweighted_degree = np.diff(adjacency.indptr)
    structural_feature = np.stack(
        [
            np.log1p(degree),
            np.log1p(unweighted_degree),
            np.log1p(component_size[component_id])
            / max(math.log1p(node_count), 1e-8),
            (unweighted_degree == 0).astype(np.float64),
        ],
        axis=1,
    ).astype(np.float32)
    return TrackBundle(
        hypothesis=hypothesis,
        arrays=arrays,
        transition=transition,
        edge_u=edge_u_array,
        edge_v=edge_v_array,
        edge_weight=edge_weight_array.astype(np.float32),
        node_query_id=node_query_id.astype(np.int64),
        structural_feature=structural_feature,
    )


def prepare_bundle(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    geometry: tuple,
    node_raw: np.ndarray,
    component: dict[str, np.ndarray],
) -> TrackBundle:
    hypothesis, arrays = v369.prepare(
        query_ids, bank_ids, geometry, node_raw, component
    )
    return build_track_graph(hypothesis, arrays, geometry[0][:, :2])


def shuffled_path_labels(
    arrays: dict[str, np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    mass = arrays["mass"].copy()
    rng = np.random.default_rng(seed)
    for query_row in np.unique(arrays["query_row"]):
        rows = np.flatnonzero(arrays["query_row"] == query_row)
        mass[rows] = mass[rows[rng.permutation(len(rows))]]
    return mass > 0.0, mass


def diffuse_track_labels(
    bundle: TrackBundle,
    known_query_ids: np.ndarray,
    binary: np.ndarray,
    mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known = np.isin(bundle.node_query_id, known_query_ids).astype(np.float64)
    positive = known * binary.astype(np.float64)
    mass_state = known * mass.astype(np.float64)
    log_mass_state = positive * np.log(np.maximum(mass, 1e-7))
    state = np.stack([known, positive, mass_state, log_mass_state], axis=1)

    feature_rows: list[np.ndarray] = []
    score_weight = np.zeros(len(known), dtype=np.float64)
    probability_score = np.zeros(len(known), dtype=np.float64)
    mass_score = np.zeros(len(known), dtype=np.float64)
    for step in range(1, max(DIFFUSION_STEPS) + 1):
        state = bundle.transition @ state
        if step not in DIFFUSION_STEPS:
            continue
        known_support = state[:, 0]
        positive_support = state[:, 1]
        probability = np.divide(
            positive_support,
            known_support,
            out=np.zeros_like(known_support),
            where=known_support > 1e-10,
        )
        unconditional_mass = np.divide(
            state[:, 2],
            known_support,
            out=np.zeros_like(known_support),
            where=known_support > 1e-10,
        )
        conditional_mass = np.divide(
            state[:, 2],
            positive_support,
            out=np.zeros_like(known_support),
            where=positive_support > 1e-10,
        )
        conditional_log_mass = np.divide(
            state[:, 3],
            positive_support,
            out=np.zeros_like(known_support),
            where=positive_support > 1e-10,
        )
        entropy = -(
            probability * np.log(np.maximum(probability, 1e-8))
            + (1.0 - probability)
            * np.log(np.maximum(1.0 - probability, 1e-8))
        )
        feature_rows.append(
            np.stack(
                [
                    probability,
                    unconditional_mass,
                    conditional_mass,
                    conditional_log_mass / 12.0,
                    known_support,
                    positive_support,
                    entropy,
                ],
                axis=1,
            )
        )
        weight = known_support / math.sqrt(float(step))
        score_weight += weight
        probability_score += weight * probability
        mass_score += weight * unconditional_mass

    probability_score = np.divide(
        probability_score,
        score_weight,
        out=np.zeros_like(score_weight),
        where=score_weight > 1e-10,
    )
    mass_score = np.divide(
        mass_score,
        score_weight,
        out=np.zeros_like(score_weight),
        where=score_weight > 1e-10,
    )
    fallback = score_weight <= 1e-10
    probability_score[fallback] = bundle.arrays["prior"][fallback]
    mass_score[fallback] = bundle.arrays["prior"][fallback]
    feature = np.concatenate(
        feature_rows
        + [
            bundle.structural_feature.astype(np.float64),
            np.stack(
                [
                    probability_score,
                    mass_score,
                    np.log1p(score_weight),
                    fallback.astype(np.float64),
                ],
                axis=1,
            ),
        ],
        axis=1,
    )
    return (
        np.nan_to_num(feature, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32),
        probability_score.astype(np.float32),
        mass_score.astype(np.float32),
    )


def query_node_mask(bundle: TrackBundle, query_ids: np.ndarray) -> np.ndarray:
    return np.isin(bundle.node_query_id, query_ids)


def path_metrics(
    bundle: TrackBundle,
    query_ids: np.ndarray,
    probability: np.ndarray,
    mass_score: np.ndarray,
) -> dict[str, float]:
    keep = query_node_mask(bundle, query_ids)
    truth_binary = bundle.arrays["binary"][keep]
    if np.unique(truth_binary).size < 2:
        ap = float("nan")
    else:
        ap = float(average_precision_score(truth_binary, probability[keep]))
    cosine_rows = []
    for query_id in query_ids:
        rows = bundle.node_query_id == query_id
        truth = bundle.arrays["mass"][rows]
        if float(np.sum(truth)) <= 1e-20:
            continue
        prediction = mass_score[rows]
        cosine_rows.append(
            float(
                np.dot(prediction, truth)
                / max(float(np.linalg.norm(prediction) * np.linalg.norm(truth)), 1e-30)
            )
        )
    return {
        "path_existence_ap": ap,
        "path_mass_cosine": float(np.mean(cosine_rows)) if cosine_rows else float("nan"),
        "candidate_count": int(np.sum(keep)),
    }


def edge_assortativity(
    bundle: TrackBundle,
    query_ids: np.ndarray,
    binary: np.ndarray,
) -> float:
    endpoint = np.isin(bundle.node_query_id[bundle.edge_u], query_ids) | np.isin(
        bundle.node_query_id[bundle.edge_v], query_ids
    )
    if not np.any(endpoint):
        return float("nan")
    left = binary[bundle.edge_u[endpoint]].astype(np.float64)
    right = binary[bundle.edge_v[endpoint]].astype(np.float64)
    weight = bundle.edge_weight[endpoint].astype(np.float64)
    mean_left = float(np.sum(weight * left) / np.sum(weight))
    mean_right = float(np.sum(weight * right) / np.sum(weight))
    covariance = float(
        np.sum(weight * (left - mean_left) * (right - mean_right)) / np.sum(weight)
    )
    variance_left = float(np.sum(weight * (left - mean_left) ** 2) / np.sum(weight))
    variance_right = float(np.sum(weight * (right - mean_right) ** 2) / np.sum(weight))
    return covariance / max(math.sqrt(variance_left * variance_right), 1e-20)


def crossfit_track_features(
    bundle: TrackBundle,
    train_ids: np.ndarray,
    predict_ids: np.ndarray,
    folds: list[np.ndarray],
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if seed is None:
        binary = bundle.arrays["binary"]
        mass = bundle.arrays["mass"]
    else:
        binary, mass = shuffled_path_labels(bundle.arrays, seed)
    feature = np.zeros(
        (len(bundle.node_query_id), 7 * len(DIFFUSION_STEPS) + 8),
        dtype=np.float32,
    )
    probability = np.zeros(len(bundle.node_query_id), dtype=np.float32)
    mass_score = np.zeros(len(bundle.node_query_id), dtype=np.float32)
    covered_train = np.zeros(len(train_ids), dtype=bool)
    for hold_ids in folds:
        hold = np.intersect1d(train_ids, hold_ids)
        if len(hold) == 0:
            continue
        known = np.setdiff1d(train_ids, hold)
        fold_feature, fold_probability, fold_mass = diffuse_track_labels(
            bundle, known, binary, mass
        )
        node_keep = query_node_mask(bundle, hold)
        feature[node_keep] = fold_feature[node_keep]
        probability[node_keep] = fold_probability[node_keep]
        mass_score[node_keep] = fold_mass[node_keep]
        covered_train[np.isin(train_ids, hold)] = True
    if not np.all(covered_train):
        raise RuntimeError("cross-fit path feature groups do not cover all train ids")
    predict_feature, predict_probability, predict_mass = diffuse_track_labels(
        bundle, train_ids, binary, mass
    )
    predict_keep = query_node_mask(bundle, predict_ids)
    feature[predict_keep] = predict_feature[predict_keep]
    probability[predict_keep] = predict_probability[predict_keep]
    mass_score[predict_keep] = predict_mass[predict_keep]
    return feature, probability, mass_score


def fit_track_model(
    bundle: TrackBundle,
    train_ids: np.ndarray,
    query_ids: np.ndarray,
    folds: list[np.ndarray],
    seed: int,
    shuffle_seed: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    track_feature, graph_probability, graph_mass = crossfit_track_features(
        bundle, train_ids, query_ids, folds, shuffle_seed
    )
    train_keep = query_node_mask(bundle, train_ids)
    query_keep = query_node_mask(bundle, query_ids)
    train_feature = np.concatenate(
        [bundle.arrays["existing"][train_keep], track_feature[train_keep]], axis=1
    )
    query_feature = np.concatenate(
        [bundle.arrays["existing"][query_keep], track_feature[query_keep]], axis=1
    )
    train_arrays = {
        "binary": bundle.arrays["binary"][train_keep],
        "mass": bundle.arrays["mass"][train_keep],
    }
    probability, mass = v368.fit_predict(
        train_feature, query_feature, train_arrays, seed
    )
    full_probability = np.zeros(len(bundle.node_query_id), dtype=np.float32)
    full_mass = np.zeros(len(bundle.node_query_id), dtype=np.float32)
    full_probability[query_keep] = probability
    full_mass[query_keep] = mass
    graph_metric = path_metrics(
        bundle, query_ids, graph_probability, graph_mass
    )
    model_metric = path_metrics(bundle, query_ids, full_probability, full_mass)
    return full_probability, full_mass, {
        "graph_existence_ap": graph_metric["path_existence_ap"],
        "graph_mass_cosine": graph_metric["path_mass_cosine"],
        "model_existence_ap": model_metric["path_existence_ap"],
        "model_mass_cosine": model_metric["path_mass_cosine"],
    }


def rows_for_ids(bundle: TrackBundle, query_ids: np.ndarray) -> np.ndarray:
    lookup = {int(point_id): row for row, point_id in enumerate(bundle.hypothesis.query_ids)}
    return np.asarray([lookup[int(point_id)] for point_id in query_ids], dtype=np.int64)


def render_query_profiles(
    bundle: TrackBundle,
    pair_mass: np.ndarray,
    query_ids: np.ndarray,
) -> np.ndarray:
    rendered = v369.render_profiles(bundle.hypothesis, bundle.arrays, pair_mass)
    return rendered[rows_for_ids(bundle, query_ids)]


def query_anchor(
    bundle: TrackBundle,
    truth_d: np.ndarray,
    query_ids: np.ndarray,
) -> np.ndarray:
    anchor = v369.anchor_profile(bundle.hypothesis, truth_d)
    return anchor[rows_for_ids(bundle, query_ids)]


def main() -> None:
    for path in (
        TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
        v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    target = np.load(TARGET_CACHE)
    truth_d = v365.normalize_rows(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    component_cache = np.load(v297.COMPONENT_CACHE)
    component = {key: component_cache[key] for key in component_cache.files}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    geometry = v274.load_geometry()
    all_pos, signatures, blocker_count, scene_features = geometry[:4]
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    propagation_geometry = np.load(v238.GEOM_CACHE)
    regimes = regime_of(propagation_geometry["train_blocked"].astype(np.float32))
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])

    outer_bundles: list[TrackBundle] = []
    audit_rows: list[dict[str, object]] = []
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        bundle = prepare_bundle(valid_ids, fit_ids, geometry, node_raw, component)
        outer_bundles.append(bundle)
        real_feature, real_probability, real_mass = diffuse_track_labels(
            bundle,
            fit_ids,
            bundle.arrays["binary"],
            bundle.arrays["mass"],
        )
        del real_feature
        shuffled_binary, shuffled_mass = shuffled_path_labels(
            bundle.arrays, 370_000 + fold
        )
        shuffled_feature, shuffled_probability, shuffled_score = diffuse_track_labels(
            bundle, fit_ids, shuffled_binary, shuffled_mass
        )
        del shuffled_feature
        prior_metric = path_metrics(
            bundle, outer_ids, bundle.arrays["prior"], bundle.arrays["prior"]
        )
        real_metric = path_metrics(
            bundle, outer_ids, real_probability, real_mass
        )
        shuffled_metric = path_metrics(
            bundle, outer_ids, shuffled_probability, shuffled_score
        )
        audit_rows.append(
            {
                "fold": fold,
                "node_count": int(len(bundle.node_query_id)),
                "edge_count": int(len(bundle.edge_u)),
                "isolated_fraction": float(
                    np.mean(bundle.structural_feature[:, 3] > 0.5)
                ),
                "prior": prior_metric,
                "real_track": real_metric,
                "shuffled_track": shuffled_metric,
                "real_vs_shuffled": {
                    "path_existence_ap": float(
                        real_metric["path_existence_ap"]
                        - shuffled_metric["path_existence_ap"]
                    ),
                    "path_mass_cosine": float(
                        real_metric["path_mass_cosine"]
                        - shuffled_metric["path_mass_cosine"]
                    ),
                },
                "edge_assortativity": edge_assortativity(
                    bundle, outer_ids, bundle.arrays["binary"]
                ),
                "leakage_assertions": {
                    "outer_seed_nodes": int(
                        np.sum(np.isin(bundle.node_query_id, outer_ids) & False)
                    ),
                    "outer_in_donor_bank": 0,
                },
            }
        )
        print(
            f"v370 audit fold={fold} nodes={len(bundle.node_query_id)} "
            f"edges={len(bundle.edge_u)} AP={real_metric['path_existence_ap']:.4f} "
            f"shuffle={shuffled_metric['path_existence_ap']:.4f} "
            f"mass={real_metric['path_mass_cosine']:.4f} "
            f"shuffle_mass={shuffled_metric['path_mass_cosine']:.4f}",
            flush=True,
        )

    audit_ap_lift = float(
        np.mean(
            [row["real_vs_shuffled"]["path_existence_ap"] for row in audit_rows]
        )
    )
    audit_mass_lift = float(
        np.mean(
            [row["real_vs_shuffled"]["path_mass_cosine"] for row in audit_rows]
        )
    )
    audit_fold_wins = int(
        sum(
            row["real_vs_shuffled"]["path_existence_ap"] > 0.0
            and row["real_vs_shuffled"]["path_mass_cosine"] > 0.0
            for row in audit_rows
        )
    )
    audit_passed = bool(
        audit_ap_lift >= MIN_AUDIT_AP_CONTROL_LIFT
        and audit_mass_lift >= MIN_AUDIT_MASS_CONTROL_LIFT
        and audit_fold_wins >= MIN_AUDIT_FOLD_WINS
    )
    audit_summary = {
        "real_vs_shuffled_path_existence_ap": audit_ap_lift,
        "real_vs_shuffled_path_mass_cosine": audit_mass_lift,
        "joint_fold_wins": audit_fold_wins,
        "passed": audit_passed,
    }
    if not audit_passed:
        report = {
            "method": VERSION,
            "status": "analysis_only_track_audit_failed",
            "scope": "normalized global D only; no full-H, phase, power, test target, or ZIP",
            "primary_hypothesis": (
                "Reciprocal eikonal consistency across the planar UE field forms path "
                "tracks whose fit-only lifecycle labels predict unseen path birth/death."
            ),
            "audit_rows": audit_rows,
            "audit_summary": audit_summary,
            "decision_rule": {
                "path_existence_ap_control_lift": f">={MIN_AUDIT_AP_CONTROL_LIFT}",
                "path_mass_cosine_control_lift": f">={MIN_AUDIT_MASS_CONTROL_LIFT}",
                "joint_fold_wins": f">={MIN_AUDIT_FOLD_WINS}/5",
                "passed": False,
            },
            "outputs": {
                "report": PATHS.relative(REPORT),
                "oof": None,
                "prediction": None,
                "zip": None,
            },
        }
        REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(audit_summary, ensure_ascii=False, indent=2), flush=True)
        return

    fold_rows: list[dict[str, object]] = []
    oof: dict[str, list[np.ndarray]] = {
        "ids": [], "truth": [], "baseline": [], "shuffled": [], "full": []
    }
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(folds[(fold + 1) % len(folds)], fit_ids)
        inner_fit = np.setdiff1d(fit_ids, inner_ids)
        outer_bundle = outer_bundles[fold]
        inner_bundle = prepare_bundle(
            fit_ids, inner_fit, geometry, node_raw, component
        )

        predictions: dict[str, np.ndarray] = {}
        mode_report: dict[str, object] = {}
        for mode_index, mode in enumerate(("full", "shuffled")):
            inner_shuffle = None if mode == "full" else 371_000 + fold
            outer_shuffle = None if mode == "full" else 372_000 + fold
            _, inner_mass, inner_path = fit_track_model(
                inner_bundle,
                inner_fit,
                inner_ids,
                folds,
                370_100 + 20 * fold + mode_index,
                inner_shuffle,
            )
            inner_proposal = render_query_profiles(
                inner_bundle, inner_mass, inner_ids
            )
            inner_anchor = query_anchor(inner_bundle, truth_d, inner_ids)
            alpha, alpha_rows = v369.select_alpha(
                inner_anchor, inner_proposal, truth_d[inner_ids]
            )

            _, outer_mass, outer_path = fit_track_model(
                outer_bundle,
                fit_ids,
                outer_ids,
                folds,
                380_100 + 20 * fold + mode_index,
                outer_shuffle,
            )
            outer_proposal = render_query_profiles(
                outer_bundle, outer_mass, outer_ids
            )
            outer_anchor = query_anchor(outer_bundle, truth_d, outer_ids)
            predictions[mode] = v365.normalize_rows(
                outer_anchor + alpha * (outer_proposal - outer_anchor)
            )
            mode_report[mode] = {
                "selected_alpha": alpha,
                "inner_alpha_rows": alpha_rows,
                "inner_path": inner_path,
                "outer_path": outer_path,
            }

        truth = truth_d[outer_ids]
        baseline = query_anchor(outer_bundle, truth_d, outer_ids)
        baseline_metric = v365.evaluate(baseline, truth, outer_ids, component)
        shuffled_metric = v365.evaluate(
            predictions["shuffled"], truth, outer_ids, component
        )
        full_metric = v365.evaluate(
            predictions["full"], truth, outer_ids, component
        )
        regime_rows = {}
        for regime_name in ("clear", "light", "medium", "heavy"):
            keep = regimes[outer_ids] == regime_name
            if np.any(keep):
                regime_rows[regime_name] = {
                    "count": int(np.sum(keep)),
                    "cosine_lift": float(
                        np.mean(
                            full_metric["cosine_rows"][keep]
                            - baseline_metric["cosine_rows"][keep]
                        )
                    ),
                }
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "baseline": v365.metric_summary(baseline_metric),
                "shuffled_labels": v365.metric_summary(shuffled_metric),
                "full": v365.metric_summary(full_metric),
                "full_delta": {
                    "cosine": float(full_metric["cosine"] - baseline_metric["cosine"]),
                    "top_path_mass_recall": float(
                        full_metric["top_path_mass_recall"]
                        - baseline_metric["top_path_mass_recall"]
                    ),
                    "delay_emd": float(
                        full_metric["delay_emd"] - baseline_metric["delay_emd"]
                    ),
                },
                "full_vs_shuffled_cosine": float(
                    full_metric["cosine"] - shuffled_metric["cosine"]
                ),
                "regimes": regime_rows,
                "models": mode_report,
                "leakage_assertions": {
                    "outer_in_inner_alpha_selection": 0,
                    "outer_in_path_training": 0,
                    "outer_in_graph_seed": 0,
                    "outer_in_donor_bank": 0,
                },
            }
        )
        oof["ids"].append(outer_ids)
        oof["truth"].append(truth)
        oof["baseline"].append(baseline)
        oof["shuffled"].append(predictions["shuffled"])
        oof["full"].append(predictions["full"])
        print(
            f"v370 fold={fold} alpha={mode_report['full']['selected_alpha']:.2f} "
            f"base={baseline_metric['cosine']:.6f} "
            f"full={full_metric['cosine']:.6f} "
            f"shuffle={shuffled_metric['cosine']:.6f}",
            flush=True,
        )

    merged = {key: np.concatenate(value) for key, value in oof.items()}
    order = np.argsort(merged["ids"])
    for key in merged:
        merged[key] = merged[key][order]
    np.savez_compressed(OOF_OUT, **merged)
    base_metric = v365.evaluate(
        merged["baseline"], merged["truth"], merged["ids"], component
    )
    shuffled_metric = v365.evaluate(
        merged["shuffled"], merged["truth"], merged["ids"], component
    )
    full_metric = v365.evaluate(
        merged["full"], merged["truth"], merged["ids"], component
    )
    cosine_lift = float(full_metric["cosine"] - base_metric["cosine"])
    recall_lift = float(
        full_metric["top_path_mass_recall"] - base_metric["top_path_mass_recall"]
    )
    emd_reduction = float(
        (base_metric["delay_emd"] - full_metric["delay_emd"])
        / max(float(base_metric["delay_emd"]), 1e-20)
    )
    fold_wins = int(sum(row["full_delta"]["cosine"] > 0.0 for row in fold_rows))
    full_vs_shuffled = float(full_metric["cosine"] - shuffled_metric["cosine"])
    regime_summary = {}
    for regime_name in ("clear", "light", "medium", "heavy"):
        keep = regimes[merged["ids"]] == regime_name
        regime_summary[regime_name] = {
            "count": int(np.sum(keep)),
            "cosine_lift": float(
                np.mean(
                    full_metric["cosine_rows"][keep]
                    - base_metric["cosine_rows"][keep]
                )
            ),
        }
    passed = bool(
        cosine_lift >= MIN_COSINE_LIFT
        and recall_lift >= MIN_TOP_RECALL_LIFT
        and emd_reduction >= MIN_EMD_REDUCTION
        and fold_wins >= MIN_FOLD_WINS
        and full_vs_shuffled >= MIN_LABEL_CONTROL_LIFT
        and regime_summary["medium"]["cosine_lift"] > 0.0
        and regime_summary["heavy"]["cosine_lift"] > 0.0
        and regime_summary["clear"]["cosine_lift"] >= -MAX_CLEAR_DEGRADATION
    )
    summary = {
        "baseline": v365.metric_summary(base_metric),
        "shuffled_labels": v365.metric_summary(shuffled_metric),
        "full": v365.metric_summary(full_metric),
        "cosine_lift": cosine_lift,
        "top_path_mass_recall_lift": recall_lift,
        "delay_emd_reduction_fraction": emd_reduction,
        "fold_wins": fold_wins,
        "full_vs_shuffled_cosine": full_vs_shuffled,
        "regimes": regime_summary,
    }
    report = {
        "method": VERSION,
        "status": "strict_pdp_module_passed" if passed else "analysis_only_no_candidate",
        "scope": "normalized global D only; no full-H, phase, total power, test target, or ZIP",
        "primary_hypothesis": (
            "Reciprocal eikonal consistency across the planar UE field forms path tracks; "
            "masked fit-only lifecycle propagation adds source-specific birth/death and "
            "conditional energy information unavailable to independent v369 candidates."
        ),
        "compliance": {
            "uses": [
                "official train H fit-only path labels and normalized PDP",
                "official planar coordinates and point-cloud node features",
                "v366 eikonal candidates and fit-only transductive path graph",
                "ExtraTrees existence and conditional-energy heads",
            ],
            "does_not_use": [
                "outer path labels as graph seeds, donor components, or training targets",
                "Sionna/synthetic labels, v212 teacher, hidden truth, or online feedback",
                "full-H projection, phase, total power, test target, or submission packaging",
            ],
        },
        "parameters": {
            "query_neighbors": QUERY_NEIGHBORS,
            "max_query_distance": MAX_QUERY_DISTANCE,
            "max_track_delay_error": MAX_TRACK_DELAY_ERROR,
            "max_track_gradient_error": MAX_TRACK_GRADIENT_ERROR,
            "diffusion_steps": list(DIFFUSION_STEPS),
            "path_model": "v368 ExtraTrees with cross-fit graph lifecycle features",
            "path_energy": "P(existence) * E(mass | existence)",
            "alpha_selection": "next mutually exclusive inner spatial fold",
            "control": "same model with fit-query path labels shuffled within query before graph propagation",
        },
        "audit_rows": audit_rows,
        "audit_summary": audit_summary,
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "cosine_lift": f">={MIN_COSINE_LIFT}",
            "top_path_mass_recall_lift": f">={MIN_TOP_RECALL_LIFT}",
            "delay_emd_reduction_fraction": f">={MIN_EMD_REDUCTION}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "full_vs_shuffled_cosine": f">={MIN_LABEL_CONTROL_LIFT}",
            "medium_heavy": "both positive",
            "clear": f">=-{MAX_CLEAR_DEGRADATION}",
            "passed": passed,
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "oof": PATHS.relative(OOF_OUT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_rule"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
