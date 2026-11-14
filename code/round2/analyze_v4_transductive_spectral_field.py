#!/usr/bin/env python3
"""Audit transductive graph extension across the ten Round 2 test-like holes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from data import load_positions
from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_transductive_spectral_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_transductive_spectral_field"
REPORT = ANALYSIS_DIR / "transductive_spectral_field.json"
WEIGHT_CACHE = ARTIFACT_DIR / "oof_transductive_weights.npz"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
GEOMETRY = PRE_DIR / "round2_geometry_visibility_features.npz"
OOF_META = PRE_DIR / "round2_pre_sionna_oof_foundation_meta.npz"
FOLD_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
V2_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_oof_weights.npz"
)

GRAPH_NEIGHBORS = (12, 24, 36)
RIDGE_FRACTIONS = (0.0, 0.05, 0.20, 1.0)
BLEND_ALPHAS = (0.25, 0.50, 0.75, 1.00)
GEOMETRY_STRENGTH = 0.50
LOCAL_SCALE_NEIGHBOR = 8
TOP_WEIGHTS = 256


@dataclass(frozen=True)
class Config:
    mode: str
    neighbors: int
    ridge_fraction: float
    alpha: float

    @property
    def name(self) -> str:
        return (
            f"{self.mode}_k{self.neighbors}_r{self.ridge_fraction:g}"
            f"_a{self.alpha:g}"
        )


def normalize_power(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = output.sum(axis=1, keepdims=True)
    return np.divide(
        output,
        np.maximum(total, 1e-30),
        out=np.zeros_like(output),
        where=total > 0.0,
    )


def normalize_amplitude(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    norm = np.linalg.norm(output, axis=1, keepdims=True)
    return np.divide(
        output,
        np.maximum(norm, 1e-30),
        out=np.zeros_like(output),
        where=norm > 0.0,
    )


def row_cosine(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    numerator = np.sum(prediction * truth, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1)
    return np.divide(
        numerator,
        np.maximum(denominator, 1e-30),
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def summarize(
    pas_prediction: np.ndarray,
    pdp_prediction: np.ndarray,
    pas_truth: np.ndarray,
    pdp_truth: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
) -> dict[str, object]:
    pas = row_cosine(pas_prediction, pas_truth)
    pdp = row_cosine(pdp_prediction, pdp_truth)
    joint = 0.5 * (pas + pdp)
    return {
        "pas": float(np.mean(pas)),
        "pdp": float(np.mean(pdp)),
        "mean": float(np.mean(joint)),
        "fold": [
            float(np.mean(joint[fold_id == fold])) for fold in np.unique(fold_id)
        ],
        "site": {
            str(int(value)): float(np.mean(joint[site == value]))
            for value in np.unique(site)
        },
    }


def geometry_embedding(
    raw: np.ndarray,
    fit_count: int,
) -> np.ndarray:
    # Keep BS-relative and official-map features, while XY remains in the spatial kernel.
    selected = np.asarray(raw[:, 6:], dtype=np.float64)
    mean = np.mean(selected[:fit_count], axis=0)
    scale = np.std(selected[:fit_count], axis=0)
    standardized = np.clip((selected - mean) / np.maximum(scale, 1e-4), -8.0, 8.0)
    components = min(12, standardized.shape[1], max(fit_count - 1, 1))
    return PCA(n_components=components, random_state=20_260_805).fit(
        standardized[:fit_count]
    ).transform(standardized)


def build_graph(
    xy: np.ndarray,
    geometry: np.ndarray | None,
    neighbors: int,
) -> sparse.csr_matrix:
    take = min(neighbors + 1, len(xy))
    distance, neighbor = cKDTree(xy).query(xy, k=take, workers=-1)
    distance = np.asarray(distance[:, 1:], dtype=np.float64)
    neighbor = np.asarray(neighbor[:, 1:], dtype=np.int64)
    scale_column = min(LOCAL_SCALE_NEIGHBOR - 1, distance.shape[1] - 1)
    local_scale = np.maximum(distance[:, scale_column], 1.0)
    source = np.repeat(np.arange(len(xy), dtype=np.int64), distance.shape[1])
    target = neighbor.reshape(-1)
    denominator = np.maximum(local_scale[source] * local_scale[target], 1e-8)
    weight = np.exp(-(distance.reshape(-1) ** 2) / denominator)
    if geometry is not None:
        difference = geometry[source] - geometry[target]
        geometry_distance = np.sum(difference * difference, axis=1)
        positive = geometry_distance[geometry_distance > 1e-10]
        geometry_scale = float(np.median(positive)) if len(positive) else 1.0
        weight *= np.exp(
            -GEOMETRY_STRENGTH * geometry_distance / max(geometry_scale, 1e-8)
        )
    graph = sparse.coo_matrix(
        (weight, (source, target)), shape=(len(xy), len(xy))
    ).tocsr()
    graph = graph.maximum(graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def anchor_matrix(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    lookup = {int(point_id): row for row, point_id in enumerate(fit_ids)}
    output = np.zeros((len(query_ids), len(fit_ids)), dtype=np.float64)
    for row, (ids, values) in enumerate(zip(donor_ids, weights)):
        for point_id, value in zip(ids, values):
            local = lookup.get(int(point_id))
            if local is None:
                raise RuntimeError(f"OOF donor {int(point_id)} is not in fit-only bank")
            output[row, local] += float(value)
    return output


def proposal_matrix(
    graph: sparse.csr_matrix,
    fit_count: int,
    anchor: np.ndarray,
    ridge_fraction: float,
) -> np.ndarray:
    unknown = np.arange(fit_count, graph.shape[0], dtype=np.int64)
    known = np.arange(fit_count, dtype=np.int64)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    w_uu = graph[unknown][:, unknown]
    w_ul = graph[unknown][:, known]
    degree_scale = max(float(np.median(degree[unknown])), 1e-8)
    ridge = ridge_fraction * degree_scale
    numerical_ridge = max(1e-10 * degree_scale, 1e-12)
    system = sparse.diags(degree[unknown] + ridge + numerical_ridge) - w_uu
    right = w_ul.toarray()
    if ridge > 0.0:
        right += ridge * anchor
    solved = np.asarray(spsolve(system.tocsc(), right), dtype=np.float64)
    if solved.ndim == 1:
        solved = solved[None, :]
    row_sum = solved.sum(axis=1, keepdims=True)
    return np.divide(
        solved,
        np.maximum(row_sum, 1e-30),
        out=np.zeros_like(solved),
        where=np.abs(row_sum) > 1e-30,
    )


def top_weight_rows(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    matrix: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    take = min(count, matrix.shape[1])
    partition = np.argpartition(np.abs(matrix), -take, axis=1)[:, -take:]
    values = np.take_along_axis(matrix, partition, axis=1)
    order = np.argsort(-np.abs(values), axis=1)
    partition = np.take_along_axis(partition, order, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    total = values.sum(axis=1, keepdims=True)
    values = np.divide(
        values,
        np.maximum(total, 1e-30),
        out=np.zeros_like(values),
        where=np.abs(total) > 1e-30,
    )
    return query_ids, fit_ids[partition], values.astype(np.float32)


def topology_report(
    train_xy: np.ndarray,
    test_xy: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for site_value in np.unique(test_site):
        test_rows = np.flatnonzero(test_site == site_value)
        local = test_xy[test_rows]
        labels = DBSCAN(eps=15.0, min_samples=3).fit_predict(local)
        train_tree = cKDTree(train_xy[train_site == site_value])
        for cluster in np.unique(labels):
            points = local[labels == cluster]
            centered = points - np.mean(points, axis=0, keepdims=True)
            covariance = centered.T @ centered / max(len(points) - 1, 1)
            eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
            nearest_test = cKDTree(points).query(points, k=min(2, len(points)))[0]
            nearest_train = train_tree.query(points, k=1)[0]
            hull_area = float(ConvexHull(points).volume) if len(points) >= 3 else 0.0
            rows.append(
                {
                    "site": int(site_value),
                    "cluster": int(cluster),
                    "count": int(len(points)),
                    "pca_first_fraction": float(
                        eigenvalues[0] / max(float(np.sum(eigenvalues)), 1e-30)
                    ),
                    "nearest_test_median_m": float(np.median(nearest_test[:, -1])),
                    "nearest_train_median_m": float(np.median(nearest_train)),
                    "nearest_train_p90_m": float(np.quantile(nearest_train, 0.90)),
                    "convex_hull_area_m2": hull_area,
                }
            )
    return rows


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (DESCRIPTORS, GEOMETRY, OOF_META, FOLD_CACHE, V2_WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(path)

    _, train_positions, test_positions, train_site, test_site = load_positions()
    descriptors = np.load(DESCRIPTORS)
    pas_truth_all = normalize_power(descriptors["global_pas"])
    pdp_truth_all = normalize_power(descriptors["global_pdp"])
    pas_amplitude_all = np.sqrt(pas_truth_all)
    pdp_amplitude_all = np.sqrt(pdp_truth_all)
    valid = descriptors["valid"].astype(bool)
    geometry_cache = np.load(GEOMETRY)
    train_geometry = geometry_cache["train"].astype(np.float64)

    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    query_site = train_site[query_ids]
    fold_cache = np.load(FOLD_CACHE)
    v2 = np.load(V2_WEIGHTS)
    if not np.array_equal(v2["query_ids"], query_ids):
        raise RuntimeError("V2 OOF weight order does not match OOF metadata")

    base_pas_amplitude = np.empty((len(query_ids), pas_amplitude_all.shape[1]))
    base_pdp_amplitude = np.empty((len(query_ids), pdp_amplitude_all.shape[1]))
    for row, (ids, weights) in enumerate(zip(v2["neighbor_ids"], v2["weights"])):
        base_pas_amplitude[row] = normalize_amplitude(
            np.sum(weights[:, None] * pas_amplitude_all[ids], axis=0)[None, :]
        )[0]
        base_pdp_amplitude[row] = normalize_amplitude(
            np.sum(weights[:, None] * pdp_amplitude_all[ids], axis=0)[None, :]
        )[0]
    base_pas = np.square(base_pas_amplitude)
    base_pdp = np.square(base_pdp_amplitude)
    baseline = summarize(
        base_pas,
        base_pdp,
        pas_truth_all[query_ids],
        pdp_truth_all[query_ids],
        fold_id,
        query_site,
    )

    configs = [
        Config(mode, neighbors, ridge, alpha)
        for mode in ("xy", "geometry", "shuffled_geometry")
        for neighbors in GRAPH_NEIGHBORS
        for ridge in RIDGE_FRACTIONS
        for alpha in BLEND_ALPHAS
    ]
    prediction_pas = {
        config.name: np.empty_like(base_pas_amplitude) for config in configs
    }
    prediction_pdp = {
        config.name: np.empty_like(base_pdp_amplitude) for config in configs
    }
    matrix_cache: dict[tuple[int, int, str, int, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(20_260_805)
    shuffled_index_by_site = {
        int(site_value): rng.permutation(np.flatnonzero(train_site == site_value))
        for site_value in np.unique(train_site)
    }

    for fold in np.unique(fold_id):
        fold_rows = np.flatnonzero(fold_id == fold)
        fold_query = query_ids[fold_rows]
        fit_all = fold_cache[f"fold{int(fold)}_fit"].astype(np.int64)
        for site_value in np.unique(query_site[fold_rows]):
            output_rows = fold_rows[query_site[fold_rows] == site_value]
            query = query_ids[output_rows]
            fit = fit_all[(train_site[fit_all] == site_value) & valid[fit_all]]
            donor = v2["neighbor_ids"][output_rows].astype(np.int64)
            donor_weight = v2["weights"][output_rows].astype(np.float64)
            anchor = anchor_matrix(fit, query, donor, donor_weight)
            nodes = np.concatenate([fit, query])
            xy = train_positions[nodes, :2]
            raw_geometry = train_geometry[nodes]
            embedded = geometry_embedding(raw_geometry, len(fit))
            shuffled_global = shuffled_index_by_site[int(site_value)]
            shuffled_raw = train_geometry[
                shuffled_global[np.arange(len(nodes)) % len(shuffled_global)]
            ]
            shuffled_embedded = geometry_embedding(shuffled_raw, len(fit))
            mode_geometry = {
                "xy": None,
                "geometry": embedded,
                "shuffled_geometry": shuffled_embedded,
            }
            anchor_pas = normalize_amplitude(anchor @ pas_amplitude_all[fit])
            anchor_pdp = normalize_amplitude(anchor @ pdp_amplitude_all[fit])
            for mode, graph_geometry in mode_geometry.items():
                for neighbors in GRAPH_NEIGHBORS:
                    graph = build_graph(xy, graph_geometry, neighbors)
                    for ridge in RIDGE_FRACTIONS:
                        proposal = proposal_matrix(graph, len(fit), anchor, ridge)
                        proposal_pas = normalize_amplitude(
                            proposal @ pas_amplitude_all[fit]
                        )
                        proposal_pdp = normalize_amplitude(
                            proposal @ pdp_amplitude_all[fit]
                        )
                        for alpha in BLEND_ALPHAS:
                            config = Config(mode, neighbors, ridge, alpha)
                            prediction_pas[config.name][output_rows] = normalize_amplitude(
                                (1.0 - alpha) * anchor_pas + alpha * proposal_pas
                            )
                            prediction_pdp[config.name][output_rows] = normalize_amplitude(
                                (1.0 - alpha) * anchor_pdp + alpha * proposal_pdp
                            )
                            combined = (1.0 - alpha) * anchor + alpha * proposal
                            key = (int(fold), int(site_value), mode, neighbors, ridge, alpha)
                            matrix_cache[key] = (query, fit, combined)
            print(
                f"transductive fold={int(fold)} site={int(site_value)} "
                f"query={len(query)} fit={len(fit)}",
                flush=True,
            )

    result_rows: list[dict[str, object]] = []
    point_scores: dict[str, np.ndarray] = {}
    for config in configs:
        pas = np.square(prediction_pas[config.name])
        pdp = np.square(prediction_pdp[config.name])
        summary = summarize(
            pas,
            pdp,
            pas_truth_all[query_ids],
            pdp_truth_all[query_ids],
            fold_id,
            query_site,
        )
        pas_point = row_cosine(pas, pas_truth_all[query_ids])
        pdp_point = row_cosine(pdp, pdp_truth_all[query_ids])
        point_scores[config.name] = 0.5 * (pas_point + pdp_point)
        result_rows.append({"config": config.__dict__, "name": config.name, **summary})

    base_point = 0.5 * (
        row_cosine(base_pas, pas_truth_all[query_ids])
        + row_cosine(base_pdp, pdp_truth_all[query_ids])
    )
    nested_choices: list[dict[str, object]] = []
    nested_pas = np.empty_like(base_pas)
    nested_pdp = np.empty_like(base_pdp)
    for held_fold in np.unique(fold_id):
        train_mask = fold_id != held_fold
        held_mask = fold_id == held_fold
        eligible = [row for row in result_rows if row["config"]["mode"] != "shuffled_geometry"]
        selected = max(
            eligible,
            key=lambda row: float(np.mean(point_scores[row["name"]][train_mask])),
        )
        name = str(selected["name"])
        nested_pas[held_mask] = np.square(prediction_pas[name][held_mask])
        nested_pdp[held_mask] = np.square(prediction_pdp[name][held_mask])
        nested_choices.append(
            {
                "held_fold": int(held_fold),
                "config": selected["config"],
                "selection_mean": float(np.mean(point_scores[name][train_mask])),
                "held_mean": float(np.mean(point_scores[name][held_mask])),
                "held_delta": float(
                    np.mean(point_scores[name][held_mask] - base_point[held_mask])
                ),
            }
        )
    nested = summarize(
        nested_pas,
        nested_pdp,
        pas_truth_all[query_ids],
        pdp_truth_all[query_ids],
        fold_id,
        query_site,
    )

    geometry_rows = [row for row in result_rows if row["config"]["mode"] == "geometry"]
    best_geometry = max(geometry_rows, key=lambda row: row["mean"])
    best_config = Config(**best_geometry["config"])
    matching_xy = next(
        row
        for row in result_rows
        if row["config"]
        == {
            **best_geometry["config"],
            "mode": "xy",
        }
    )
    matching_shuffle = next(
        row
        for row in result_rows
        if row["config"]
        == {
            **best_geometry["config"],
            "mode": "shuffled_geometry",
        }
    )

    final_config = max(
        [row for row in result_rows if row["config"]["mode"] != "shuffled_geometry"],
        key=lambda row: row["mean"],
    )
    final = Config(**final_config["config"])
    saved_query: list[np.ndarray] = []
    saved_donor: list[np.ndarray] = []
    saved_weight: list[np.ndarray] = []
    for fold in np.unique(fold_id):
        for site_value in np.unique(query_site[fold_id == fold]):
            query, fit, matrix = matrix_cache[
                (
                    int(fold),
                    int(site_value),
                    final.mode,
                    final.neighbors,
                    final.ridge_fraction,
                    final.alpha,
                )
            ]
            block_query, block_donor, block_weight = top_weight_rows(
                query, fit, matrix, TOP_WEIGHTS
            )
            saved_query.append(block_query)
            saved_donor.append(block_donor)
            saved_weight.append(block_weight)
    merged_query = np.concatenate(saved_query)
    order = np.argsort(merged_query)
    np.savez_compressed(
        WEIGHT_CACHE,
        query_ids=merged_query[order].astype(np.int32),
        donor_ids=np.concatenate(saved_donor)[order].astype(np.int32),
        weights=np.concatenate(saved_weight)[order].astype(np.float32),
        mode=np.asarray(final.mode),
        neighbors=np.int32(final.neighbors),
        ridge_fraction=np.float32(final.ridge_fraction),
        alpha=np.float32(final.alpha),
    )

    ranked = sorted(result_rows, key=lambda row: row["mean"], reverse=True)
    report = {
        "experiment": "Round2-E10-transductive-spectral-field",
        "status": "global_spectrum_screen_complete",
        "primary_hypothesis": (
            "Label-clamped propagation over the complete unknown test-hole graph "
            "uses test-test continuity and both boundaries, improving over independent V2 RBF queries."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "validation": "fixed 500-point five-fold spatial holes; held labels never enter propagation",
        "parameters": {
            "graph_neighbors": list(GRAPH_NEIGHBORS),
            "ridge_fractions": list(RIDGE_FRACTIONS),
            "blend_alphas": list(BLEND_ALPHAS),
            "geometry_strength": GEOMETRY_STRENGTH,
            "top_weights": TOP_WEIGHTS,
        },
        "test_topology": topology_report(
            train_positions[:, :2],
            test_positions[:, :2],
            train_site,
            test_site,
        ),
        "baseline_v2_global": baseline,
        "nested_selected": nested,
        "nested_choices": nested_choices,
        "best_geometry": best_geometry,
        "matched_controls": {
            "xy": matching_xy,
            "shuffled_geometry": matching_shuffle,
            "geometry_vs_xy": float(best_geometry["mean"] - matching_xy["mean"]),
            "geometry_vs_shuffled": float(
                best_geometry["mean"] - matching_shuffle["mean"]
            ),
        },
        "best_non_shuffled": final_config,
        "top_ten": ranked[:10],
        "outputs": {
            "weights": PATHS.relative(WEIGHT_CACHE),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
