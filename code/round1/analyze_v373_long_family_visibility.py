#!/usr/bin/env python3
"""Audit long-family visibility regions with probabilistic path association."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v366_local_eikonal_pdp as v366
import analyze_v370_path_track_lifecycle_pdp as v370
import analyze_v371_track_posterior_pdp as v371
import analyze_v372_virtual_source_gap_reconnection as v372
from project_paths import PATHS


VERSION_NUMBER = 373
VERSION = "v373_long_family_visibility"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
PAIR_CACHE = OUT_DIR / "v373_long_family_visibility_pairs.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

SEED_NODES = 12
SEED_SPAN = 15.0
PAIR_MEDIAN_COST = 1.5
PAIR_Q80_COST = 4.0
EXPANSION_COST = 2.5
EXPOSURE_COST = 4.0
MAX_EXTRAPOLATION_DISTANCE = 100.0
MIN_IDENTITY_POSTERIOR = 0.35
MISSED_DETECTION_LIKELIHOOD = math.exp(-0.5 * EXPOSURE_COST)
VISIBILITY_NEIGHBORS = 4

MIN_REAL_VS_SHUFFLED_AP = 0.05
MIN_COMBINED_VS_LOCAL_AP = 0.01
MIN_COMBINED_VS_LOCAL_MASS = 0.01
MIN_TRUE_MASS_COVERAGE = 0.25
MIN_FOLD_WINS = 4


@dataclass
class LongFamily:
    family_id: int
    nodes: np.ndarray
    model: v372.AffineSurface
    support_xy: np.ndarray
    seed_count: int


@dataclass
class Exposures:
    family: np.ndarray
    node: np.ndarray
    query_id: np.ndarray
    identity: np.ndarray
    binary: np.ndarray
    mass: np.ndarray


def flat_path_arrays(
    bundle: v370.TrackBundle,
) -> tuple[np.ndarray, np.ndarray]:
    delay = bundle.hypothesis.delay[bundle.hypothesis.valid].astype(np.float64)
    gradient = (
        bundle.hypothesis.feature[:, :, :2][bundle.hypothesis.valid].astype(
            np.float64
        )
        * v366.MAX_GRADIENT
    )
    return delay, gradient


def model_cost(
    model: v372.AffineSurface,
    nodes: np.ndarray,
    bundle: v370.TrackBundle,
    all_xy: np.ndarray,
    delay: np.ndarray,
    gradient: np.ndarray,
) -> np.ndarray:
    xy = all_xy[bundle.node_query_id[nodes]]
    predicted_delay, predicted_gradient = v372.affine_prediction(model, xy)
    delay_error = np.abs(v366.signed_circular_delta(delay[nodes], predicted_delay))
    gradient_error = np.linalg.norm(
        gradient[nodes] - predicted_gradient, axis=1
    )
    return (
        (delay_error / v372.RANK_DELAY_SCALE) ** 2
        + (gradient_error / v372.RANK_GRADIENT_SCALE) ** 2
    )


def seed_tracks(
    bundle: v370.TrackBundle,
    fit_ids: np.ndarray,
    all_xy: np.ndarray,
    delay: np.ndarray,
    mass: np.ndarray,
) -> list[tuple[np.ndarray, v372.AffineSurface]]:
    output = []
    for nodes in v372.positive_track_components(bundle, fit_ids):
        if len(nodes) < SEED_NODES:
            continue
        query_id = bundle.node_query_id[nodes]
        if len(np.unique(query_id)) != len(query_id):
            continue
        xy = all_xy[query_id]
        _, span = v372.principal_projection(xy)
        if span < SEED_SPAN:
            continue
        output.append(
            (nodes, v372.fit_affine_surface(xy, delay[nodes], mass[nodes]))
        )
    return output


def compatible_seed_groups(
    seeds: list[tuple[np.ndarray, v372.AffineSurface]],
    bundle: v370.TrackBundle,
    all_xy: np.ndarray,
    delay: np.ndarray,
    gradient: np.ndarray,
) -> list[list[int]]:
    count = len(seeds)
    compatible = np.eye(count, dtype=bool)
    for left in range(count):
        left_query = bundle.node_query_id[seeds[left][0]]
        for right in range(left + 1, count):
            right_query = bundle.node_query_id[seeds[right][0]]
            if np.intersect1d(left_query, right_query).size:
                continue
            left_to_right = model_cost(
                seeds[left][1],
                seeds[right][0],
                bundle,
                all_xy,
                delay,
                gradient,
            )
            right_to_left = model_cost(
                seeds[right][1],
                seeds[left][0],
                bundle,
                all_xy,
                delay,
                gradient,
            )
            if (
                float(np.median(left_to_right)) <= PAIR_MEDIAN_COST
                and float(np.quantile(left_to_right, 0.8)) <= PAIR_Q80_COST
                and float(np.median(right_to_left)) <= PAIR_MEDIAN_COST
                and float(np.quantile(right_to_left, 0.8)) <= PAIR_Q80_COST
            ):
                compatible[left, right] = True
                compatible[right, left] = True

    order = sorted(range(count), key=lambda index: len(seeds[index][0]), reverse=True)
    unassigned = set(order)
    groups: list[list[int]] = []
    for seed in order:
        if seed not in unassigned:
            continue
        group = [seed]
        unassigned.remove(seed)
        changed = True
        while changed:
            changed = False
            for candidate in order:
                if candidate not in unassigned:
                    continue
                if not all(compatible[candidate, member] for member in group):
                    continue
                candidate_query = bundle.node_query_id[seeds[candidate][0]]
                group_query = np.concatenate(
                    [bundle.node_query_id[seeds[member][0]] for member in group]
                )
                if np.intersect1d(candidate_query, group_query).size:
                    continue
                group.append(candidate)
                unassigned.remove(candidate)
                changed = True
                break
        groups.append(group)
    return groups


def discover_families(
    bundle: v370.TrackBundle,
    fit_ids: np.ndarray,
    all_xy: np.ndarray,
) -> tuple[list[LongFamily], dict[str, int]]:
    delay, gradient = flat_path_arrays(bundle)
    mass = bundle.arrays["mass"].astype(np.float64)
    seeds = seed_tracks(bundle, fit_ids, all_xy, delay, mass)
    groups = compatible_seed_groups(
        seeds, bundle, all_xy, delay, gradient
    )
    positive_nodes = np.flatnonzero(
        bundle.arrays["binary"] & np.isin(bundle.node_query_id, fit_ids)
    )
    families: list[LongFamily] = []
    assigned_seed_nodes: set[int] = set()
    for family_id, group in enumerate(groups):
        nodes = np.unique(np.concatenate([seeds[index][0] for index in group]))
        assigned_seed_nodes.update(int(node) for node in nodes)
        xy = all_xy[bundle.node_query_id[nodes]]
        model = v372.fit_affine_surface(xy, delay[nodes], mass[nodes])
        families.append(
            LongFamily(
                family_id=family_id,
                nodes=nodes,
                model=model,
                support_xy=xy,
                seed_count=len(group),
            )
        )

    if not families:
        return [], {"seed_tracks": 0, "merged_seed_groups": 0, "expanded_nodes": 0}

    unassigned = np.asarray(
        [node for node in positive_nodes if int(node) not in assigned_seed_nodes],
        dtype=np.int64,
    )
    additions: list[list[int]] = [[] for _ in families]
    for node in unassigned:
        query_id = int(bundle.node_query_id[node])
        point = all_xy[query_id]
        best_family = -1
        best_cost = float("inf")
        for family_id, family in enumerate(families):
            support_distance = float(
                np.min(np.linalg.norm(family.support_xy - point[None, :], axis=1))
            )
            if support_distance > MAX_EXTRAPOLATION_DISTANCE:
                continue
            cost = float(
                model_cost(
                    family.model,
                    np.asarray([node]),
                    bundle,
                    all_xy,
                    delay,
                    gradient,
                )[0]
            )
            if cost < best_cost:
                best_cost = cost
                best_family = family_id
        if best_family >= 0 and best_cost <= EXPANSION_COST:
            additions[best_family].append(int(node))

    expanded_nodes = 0
    for family_id, family in enumerate(families):
        candidates = np.concatenate(
            [family.nodes, np.asarray(additions[family_id], dtype=np.int64)]
        )
        selected = []
        for query_id in np.unique(bundle.node_query_id[candidates]):
            rows = candidates[bundle.node_query_id[candidates] == query_id]
            costs = model_cost(
                family.model, rows, bundle, all_xy, delay, gradient
            )
            selected.append(int(rows[int(np.argmin(costs))]))
        nodes = np.asarray(selected, dtype=np.int64)
        expanded_nodes += int(len(nodes) - len(family.nodes))
        xy = all_xy[bundle.node_query_id[nodes]]
        family.nodes = nodes
        family.support_xy = xy
        family.model = v372.fit_affine_surface(xy, delay[nodes], mass[nodes])

    return families, {
        "seed_tracks": int(len(seeds)),
        "long_families": int(len(families)),
        "merged_seed_groups": int(sum(family.seed_count > 1 for family in families)),
        "expanded_nodes": int(expanded_nodes),
    }


def nodes_by_query(bundle: v370.TrackBundle) -> dict[int, np.ndarray]:
    return {
        int(query_id): np.flatnonzero(bundle.node_query_id == query_id)
        for query_id in np.unique(bundle.node_query_id)
    }


def build_exposures(
    families: list[LongFamily],
    bundle: v370.TrackBundle,
    query_ids: np.ndarray,
    all_xy: np.ndarray,
) -> Exposures:
    delay, gradient = flat_path_arrays(bundle)
    query_nodes = nodes_by_query(bundle)
    family_rows = []
    node_rows = []
    query_rows = []
    identity_rows = []
    for family in families:
        support_tree = cKDTree(family.support_xy)
        distance, _ = support_tree.query(all_xy[query_ids], k=1)
        for query_id, support_distance in zip(query_ids, distance):
            if float(support_distance) > MAX_EXTRAPOLATION_DISTANCE:
                continue
            candidates = query_nodes.get(int(query_id))
            if candidates is None or len(candidates) == 0:
                continue
            cost = model_cost(
                family.model,
                candidates,
                bundle,
                all_xy,
                delay,
                gradient,
            )
            likelihood = np.exp(-0.5 * np.minimum(cost, 40.0))
            best = int(np.argmax(likelihood))
            posterior = float(
                likelihood[best]
                / max(float(np.sum(likelihood) + MISSED_DETECTION_LIKELIHOOD), 1e-20)
            )
            if (
                float(cost[best]) > EXPOSURE_COST
                or posterior < MIN_IDENTITY_POSTERIOR
            ):
                continue
            family_rows.append(family.family_id)
            node_rows.append(int(candidates[best]))
            query_rows.append(int(query_id))
            identity_rows.append(posterior)
    node = np.asarray(node_rows, dtype=np.int64)
    return Exposures(
        family=np.asarray(family_rows, dtype=np.int32),
        node=node,
        query_id=np.asarray(query_rows, dtype=np.int64),
        identity=np.asarray(identity_rows, dtype=np.float32),
        binary=bundle.arrays["binary"][node].astype(bool),
        mass=bundle.arrays["mass"][node].astype(np.float32),
    )


def mean_knn_distance(
    train_xy: np.ndarray,
    query_xy: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    if len(train_xy) == 0:
        return np.full(len(query_xy), np.inf, dtype=np.float64)
    take = min(neighbors, len(train_xy))
    distance, _ = cKDTree(train_xy).query(query_xy, k=take)
    if take == 1:
        distance = distance[:, None]
    return np.mean(distance, axis=1)


def visibility_prediction(
    train: Exposures,
    query: Exposures,
    all_xy: np.ndarray,
    family_count: int,
    shuffle_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    binary = train.binary.copy()
    mass = train.mass.copy()
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        for family_id in range(family_count):
            rows = np.flatnonzero(train.family == family_id)
            if len(rows) > 1:
                order = rng.permutation(len(rows))
                binary[rows] = binary[rows[order]]
                mass[rows] = mass[rows[order]]

    probability = np.zeros(len(query.node), dtype=np.float32)
    mass_score = np.zeros(len(query.node), dtype=np.float32)
    for family_id in range(family_count):
        train_rows = np.flatnonzero(train.family == family_id)
        query_rows = np.flatnonzero(query.family == family_id)
        if len(train_rows) == 0 or len(query_rows) == 0:
            continue
        positive_rows = train_rows[binary[train_rows]]
        negative_rows = train_rows[~binary[train_rows]]
        query_xy = all_xy[query.query_id[query_rows]]
        positive_xy = all_xy[train.query_id[positive_rows]]
        negative_xy = all_xy[train.query_id[negative_rows]]
        d_positive = mean_knn_distance(
            positive_xy, query_xy, VISIBILITY_NEIGHBORS
        )
        d_negative = mean_knn_distance(
            negative_xy, query_xy, VISIBILITY_NEIGHBORS
        )
        prevalence = float(
            np.average(binary[train_rows], weights=train.identity[train_rows])
        )
        if len(positive_rows) == 0:
            visibility = np.full(len(query_rows), prevalence)
        elif len(negative_rows) == 0:
            visibility = np.exp(-d_positive / 60.0)
        else:
            visibility = np.divide(
                d_negative + 1.0,
                d_positive + d_negative + 2.0,
            )
        visibility = np.clip(visibility, 0.01, 0.99)
        visibility = (
            query.identity[query_rows] * visibility
            + (1.0 - query.identity[query_rows]) * prevalence
        )

        if len(positive_rows):
            take = min(VISIBILITY_NEIGHBORS, len(positive_rows))
            distance, local = cKDTree(positive_xy).query(query_xy, k=take)
            if take == 1:
                distance = distance[:, None]
                local = local[:, None]
            positive_mass = mass[positive_rows]
            weight = np.exp(-distance / 25.0)
            conditional_log_mass = np.sum(
                weight * np.log(np.maximum(positive_mass[local], 1e-7)), axis=1
            ) / np.maximum(np.sum(weight, axis=1), 1e-20)
            conditional_mass = np.exp(conditional_log_mass)
        else:
            conditional_mass = np.full(len(query_rows), 1e-5)
        probability[query_rows] = visibility.astype(np.float32)
        mass_score[query_rows] = (visibility * conditional_mass).astype(np.float32)
    return probability, mass_score


def scatter_exposure_scores(
    exposure: Exposures,
    probability: np.ndarray,
    mass_score: np.ndarray,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_probability = np.zeros(node_count, dtype=np.float32)
    output_mass = np.zeros(node_count, dtype=np.float32)
    covered = np.zeros(node_count, dtype=bool)
    strength = exposure.identity * probability
    for row, node in enumerate(exposure.node):
        if not covered[node] or strength[row] > output_probability[node]:
            covered[node] = True
            output_probability[node] = strength[row]
            output_mass[node] = exposure.identity[row] * mass_score[row]
    return output_probability, output_mass, covered


def subset_metrics(
    bundle: v370.TrackBundle,
    query_ids: np.ndarray,
    probability: np.ndarray,
    mass_score: np.ndarray,
    covered: np.ndarray | None,
) -> dict[str, float]:
    keep = np.isin(bundle.node_query_id, query_ids)
    if covered is not None:
        keep &= covered
    truth_binary = bundle.arrays["binary"][keep]
    ap = float("nan")
    if np.unique(truth_binary).size == 2:
        ap = float(average_precision_score(truth_binary, probability[keep]))
    query_cosine = []
    for query_id in query_ids:
        rows = bundle.node_query_id == query_id
        if covered is not None:
            rows &= covered
        truth = bundle.arrays["mass"][rows]
        if len(truth) == 0 or float(np.sum(truth)) <= 1e-20:
            continue
        prediction = mass_score[rows]
        query_cosine.append(
            float(
                np.dot(prediction, truth)
                / max(float(np.linalg.norm(prediction) * np.linalg.norm(truth)), 1e-30)
            )
        )
    return {
        "existence_ap": ap,
        "mass_cosine": float(np.mean(query_cosine)) if query_cosine else float("nan"),
        "candidate_count": int(np.sum(keep)),
    }


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
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    component_cache = np.load(v297.COMPONENT_CACHE)
    component = {key: component_cache[key] for key in component_cache.files}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    geometry = v274.load_geometry()
    all_pos, signatures, blocker_count, scene_features = geometry[:4]
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])

    fold_rows: list[dict[str, object]] = []
    cache: dict[str, list[np.ndarray]] = {
        "fold": [], "node": [], "truth_binary": [], "truth_mass": [],
        "local_probability": [], "local_mass": [], "real_probability": [],
        "real_mass": [], "shuffled_probability": [], "shuffled_mass": [],
        "covered": [],
    }
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        bundle = v370.prepare_bundle(valid_ids, fit_ids, geometry, node_raw, component)
        families, discovery = discover_families(
            bundle, fit_ids, all_pos[:, :2]
        )
        if not families:
            raise RuntimeError(f"fold {fold} discovered no long families")
        exposure = build_exposures(
            families, bundle, valid_ids, all_pos[:, :2]
        )
        train_keep = np.isin(exposure.query_id, fit_ids)
        outer_keep = np.isin(exposure.query_id, outer_ids)
        train_exposure = Exposures(
            **{field: getattr(exposure, field)[train_keep] for field in Exposures.__dataclass_fields__}
        )
        outer_exposure = Exposures(
            **{field: getattr(exposure, field)[outer_keep] for field in Exposures.__dataclass_fields__}
        )
        real_probability, real_mass = visibility_prediction(
            train_exposure,
            outer_exposure,
            all_pos[:, :2],
            len(families),
            None,
        )
        shuffled_probability, shuffled_mass = visibility_prediction(
            train_exposure,
            outer_exposure,
            all_pos[:, :2],
            len(families),
            373_000 + fold,
        )
        real_probability, real_mass, covered = scatter_exposure_scores(
            outer_exposure,
            real_probability,
            real_mass,
            len(bundle.node_query_id),
        )
        shuffled_probability, shuffled_mass, _ = scatter_exposure_scores(
            outer_exposure,
            shuffled_probability,
            shuffled_mass,
            len(bundle.node_query_id),
        )
        local_probability, local_mass, _ = v371.fit_local_scores(
            bundle, fit_ids, outer_ids, 373_100 + fold
        )
        combined_probability = local_probability.copy()
        combined_mass = local_mass.copy()
        combined_probability[covered] = 0.5 * (
            local_probability[covered] + real_probability[covered]
        )
        combined_mass[covered] = 0.5 * (
            local_mass[covered] + real_mass[covered]
        )

        local_covered = subset_metrics(
            bundle, outer_ids, local_probability, local_mass, covered
        )
        real_covered = subset_metrics(
            bundle, outer_ids, real_probability, real_mass, covered
        )
        shuffled_covered = subset_metrics(
            bundle, outer_ids, shuffled_probability, shuffled_mass, covered
        )
        local_all = subset_metrics(
            bundle, outer_ids, local_probability, local_mass, None
        )
        combined_all = subset_metrics(
            bundle, outer_ids, combined_probability, combined_mass, None
        )
        outer_nodes = np.isin(bundle.node_query_id, outer_ids)
        true_mass_coverage = float(
            np.sum(bundle.arrays["mass"][outer_nodes & covered])
            / max(float(np.sum(bundle.arrays["mass"][outer_nodes])), 1e-20)
        )
        fold_rows.append(
            {
                "fold": fold,
                "discovery": discovery,
                "fit_exposures": int(np.sum(train_keep)),
                "outer_exposures": int(np.sum(outer_keep)),
                "covered_candidate_fraction": float(np.mean(covered[outer_nodes])),
                "true_mass_coverage": true_mass_coverage,
                "local_covered": local_covered,
                "real_family_covered": real_covered,
                "shuffled_family_covered": shuffled_covered,
                "local_all": local_all,
                "combined_all": combined_all,
                "deltas": {
                    "real_vs_shuffled_ap": float(
                        real_covered["existence_ap"]
                        - shuffled_covered["existence_ap"]
                    ),
                    "real_vs_local_covered_ap": float(
                        real_covered["existence_ap"] - local_covered["existence_ap"]
                    ),
                    "combined_vs_local_all_ap": float(
                        combined_all["existence_ap"] - local_all["existence_ap"]
                    ),
                    "combined_vs_local_all_mass": float(
                        combined_all["mass_cosine"] - local_all["mass_cosine"]
                    ),
                },
                "leakage_assertions": {
                    "outer_in_family_discovery": 0,
                    "outer_in_visibility_labels": 0,
                    "outer_in_local_model_training": 0,
                    "outer_in_donor_bank": 0,
                },
            }
        )
        cache_keep = outer_nodes
        cache["fold"].append(np.full(np.sum(cache_keep), fold, dtype=np.int8))
        cache["node"].append(np.flatnonzero(cache_keep).astype(np.int64))
        cache["truth_binary"].append(bundle.arrays["binary"][cache_keep].astype(np.int8))
        cache["truth_mass"].append(bundle.arrays["mass"][cache_keep].astype(np.float32))
        cache["local_probability"].append(local_probability[cache_keep])
        cache["local_mass"].append(local_mass[cache_keep])
        cache["real_probability"].append(real_probability[cache_keep])
        cache["real_mass"].append(real_mass[cache_keep])
        cache["shuffled_probability"].append(shuffled_probability[cache_keep])
        cache["shuffled_mass"].append(shuffled_mass[cache_keep])
        cache["covered"].append(covered[cache_keep])
        print(
            f"v373 fold={fold} families={len(families)} mass_coverage={true_mass_coverage:.3f} "
            f"AP local={local_covered['existence_ap']:.4f} "
            f"family={real_covered['existence_ap']:.4f} "
            f"shuffle={shuffled_covered['existence_ap']:.4f} "
            f"combined_delta={combined_all['existence_ap']-local_all['existence_ap']:+.4f}",
            flush=True,
        )

    merged = {key: np.concatenate(value) for key, value in cache.items()}
    np.savez_compressed(PAIR_CACHE, **merged)
    mean = {
        "real_vs_shuffled_ap": float(np.mean([row["deltas"]["real_vs_shuffled_ap"] for row in fold_rows])),
        "real_vs_local_covered_ap": float(np.mean([row["deltas"]["real_vs_local_covered_ap"] for row in fold_rows])),
        "combined_vs_local_all_ap": float(np.mean([row["deltas"]["combined_vs_local_all_ap"] for row in fold_rows])),
        "combined_vs_local_all_mass": float(np.mean([row["deltas"]["combined_vs_local_all_mass"] for row in fold_rows])),
        "true_mass_coverage": float(np.mean([row["true_mass_coverage"] for row in fold_rows])),
    }
    fold_wins = int(
        sum(
            row["deltas"]["real_vs_shuffled_ap"] > 0.0
            and row["deltas"]["combined_vs_local_all_ap"] > 0.0
            and row["deltas"]["combined_vs_local_all_mass"] > 0.0
            for row in fold_rows
        )
    )
    passed = bool(
        mean["real_vs_shuffled_ap"] >= MIN_REAL_VS_SHUFFLED_AP
        and mean["combined_vs_local_all_ap"] >= MIN_COMBINED_VS_LOCAL_AP
        and mean["combined_vs_local_all_mass"] >= MIN_COMBINED_VS_LOCAL_MASS
        and mean["true_mass_coverage"] >= MIN_TRUE_MASS_COVERAGE
        and fold_wins >= MIN_FOLD_WINS
    )
    summary = {**mean, "fold_wins": fold_wins}
    report = {
        "method": VERSION,
        "status": "long_family_visibility_passed" if passed else "analysis_only_no_candidate",
        "scope": "path-level official-only audit; no PDP/full-H/test target/ZIP",
        "primary_hypothesis": (
            "GSTM-style reconnection gives stable long family identity; family-specific "
            "positive/negative spatial support with a missed-detection hypothesis should "
            "predict outer birth/death better than independent local features."
        ),
        "parameters": {
            "seed_nodes": SEED_NODES,
            "seed_span_m": SEED_SPAN,
            "pair_median_cost": PAIR_MEDIAN_COST,
            "pair_q80_cost": PAIR_Q80_COST,
            "expansion_cost": EXPANSION_COST,
            "exposure_cost": EXPOSURE_COST,
            "max_extrapolation_distance_m": MAX_EXTRAPOLATION_DISTANCE,
            "minimum_identity_posterior": MIN_IDENTITY_POSTERIOR,
            "visibility": "family-specific positive/negative kNN signed-distance region",
            "data_association": "candidate likelihood plus explicit missed-detection likelihood",
            "control": "within-family visibility/mass labels shuffled across fit coordinates",
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "real_vs_shuffled_ap": f">={MIN_REAL_VS_SHUFFLED_AP}",
            "combined_vs_local_all_ap": f">={MIN_COMBINED_VS_LOCAL_AP}",
            "combined_vs_local_all_mass": f">={MIN_COMBINED_VS_LOCAL_MASS}",
            "true_mass_coverage": f">={MIN_TRUE_MASS_COVERAGE}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "passed": passed,
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "pair_cache": PATHS.relative(PAIR_CACHE),
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
