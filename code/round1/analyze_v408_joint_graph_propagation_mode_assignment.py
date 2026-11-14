#!/usr/bin/env python3
"""Assign whole-spectrum propagation modes jointly on a transductive UE graph."""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
import analyze_v406_neural_propagation_mode_router as v406
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 408
VERSION = "v408_joint_graph_propagation_mode_assignment"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v408_joint_graph_propagation_mode_assignment_oof.npz"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

MODE_COUNT = 32
PCA_DIM = 32
GRAPH_K = 12
GRAPH_MIX = 0.30
GRAPH_ITERATIONS = 8
ALPHAS = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float32)
CONFIDENCE_THRESHOLDS = np.asarray([0.00, 0.05, 0.10, 0.20], dtype=np.float32)
MIN_ROUTER_DELTA = 0.01
MIN_FOLD_WINS = 4
MIN_CONTROL_MARGIN = 0.002


def graph_matrix(ids: np.ndarray, xy: np.ndarray) -> sparse.csr_matrix:
    distance, local = cKDTree(xy[ids]).query(xy[ids], k=GRAPH_K + 1)
    neighbor = local[:, 1:]
    distance = distance[:, 1:]
    weight = 1.0 / np.maximum(distance, 0.25) ** 2
    rows = np.repeat(np.arange(len(ids), dtype=np.int64), GRAPH_K)
    cols = neighbor.reshape(-1)
    values = weight.reshape(-1)
    matrix = sparse.coo_matrix(
        (values, (rows, cols)), shape=(len(ids), len(ids))
    ).tocsr()
    matrix = 0.5 * (matrix + matrix.T)
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    return sparse.diags(1.0 / np.maximum(row_sum, 1e-20)) @ matrix


def direct_probabilities(
    unknown_ids: np.ndarray,
    known_ids: np.ndarray,
    known_label: np.ndarray,
    xy: np.ndarray,
) -> np.ndarray:
    distance, local = cKDTree(xy[known_ids]).query(
        xy[unknown_ids], k=min(GRAPH_K, len(known_ids))
    )
    if np.ndim(local) == 1:
        local = local[:, None]
        distance = distance[:, None]
    weight = 1.0 / np.maximum(distance, 0.25) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    probability = np.zeros((len(unknown_ids), MODE_COUNT), dtype=np.float32)
    labels = known_label[local]
    for cluster in range(MODE_COUNT):
        probability[:, cluster] = np.sum(
            weight * (labels == cluster), axis=1
        )
    return probability


def propagate_probabilities(
    all_ids: np.ndarray,
    known_ids: np.ndarray,
    known_label: np.ndarray,
    xy: np.ndarray,
    matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = np.full(len(xy), -1, dtype=np.int64)
    lookup[all_ids] = np.arange(len(all_ids))
    unknown_ids = np.setdiff1d(all_ids, known_ids)
    known_local = lookup[known_ids]
    unknown_local = lookup[unknown_ids]
    direct = direct_probabilities(unknown_ids, known_ids, known_label, xy)
    field = np.zeros((len(all_ids), MODE_COUNT), dtype=np.float32)
    field[known_local, known_label] = 1.0
    field[unknown_local] = direct
    for _ in range(GRAPH_ITERATIONS):
        updated = matrix @ field
        field[unknown_local] = (
            GRAPH_MIX * direct + (1.0 - GRAPH_MIX) * updated[unknown_local]
        )
        field[known_local] = 0.0
        field[known_local, known_label] = 1.0
    field /= np.maximum(field.sum(axis=1, keepdims=True), 1e-20)
    return direct, field[unknown_local]


def query_anchor_and_donors(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    xy: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    donor_ids, distance = v406.find_neighbors(
        query_ids, bank_ids, xy, v406.DONOR_K
    )
    anchor, _ = v406.weighted_anchor(target, donor_ids, distance)
    return anchor, donor_ids


def select_by_mode(
    probability: np.ndarray,
    donor_ids: np.ndarray,
    cluster_by_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_probability = np.take_along_axis(
        probability, cluster_by_id[donor_ids], axis=1
    )
    rank = np.argmax(candidate_probability, axis=1)
    selected_probability = candidate_probability[np.arange(len(rank)), rank]
    sorted_probability = np.sort(candidate_probability, axis=1)
    margin = selected_probability - sorted_probability[:, -2]
    confidence = 0.7 * selected_probability + 0.3 * margin
    return rank.astype(np.int64), confidence.astype(np.float32)


def render(
    anchor: np.ndarray,
    donor_ids: np.ndarray,
    rank: np.ndarray,
    confidence: np.ndarray,
    target: np.ndarray,
    alpha: float,
    threshold: float,
) -> np.ndarray:
    prediction = anchor.copy()
    active = confidence >= threshold
    rows = np.flatnonzero(active)
    donor = target[donor_ids[rows, rank[rows]]]
    prediction[active] = (1.0 - alpha) * anchor[active] + alpha * donor
    return v406.normalize(prediction)


def weighted_delta(
    prediction: np.ndarray, anchor: np.ndarray, truth: np.ndarray
) -> float:
    base = v402.point_metrics(anchor, truth)["weighted"]
    value = v402.point_metrics(prediction, truth)["weighted"]
    return float(np.mean(value - base))


def select_action(
    anchor: np.ndarray,
    donor_ids: np.ndarray,
    rank: np.ndarray,
    confidence: np.ndarray,
    target: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, float, float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best_delta = 0.0
    best_alpha = 0.0
    best_threshold = float("inf")
    for alpha in ALPHAS:
        for threshold in CONFIDENCE_THRESHOLDS:
            prediction = render(
                anchor,
                donor_ids,
                rank,
                confidence,
                target,
                float(alpha),
                float(threshold),
            )
            delta = weighted_delta(prediction, anchor, truth)
            rows.append(
                {
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "delta": delta,
                    "active_fraction": float(np.mean(confidence >= threshold)),
                }
            )
            if delta > best_delta:
                best_delta = delta
                best_alpha = float(alpha)
                best_threshold = float(threshold)
    return best_alpha, best_threshold, best_delta, rows


def evaluate(
    prediction: np.ndarray, anchor: np.ndarray, truth: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    base = v402.point_metrics(anchor, truth)
    values = v402.point_metrics(prediction, truth)
    delta = values["weighted"] - base["weighted"]
    return (
        {
            "metrics": v402.metric_summary(prediction, truth),
            "delta": {
                key: float(np.mean(values[key] - base[key])) for key in values
            },
            "point_gain": summarize(delta),
            "point_win_fraction": float(np.mean(delta > 0.0)),
        },
        delta,
    )


def fold_experiment(
    fold: int,
    folds: list[np.ndarray],
    valid_ids: np.ndarray,
    xy: np.ndarray,
    target: np.ndarray,
    blocked: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    query_ids = folds[fold]
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    validation_ids = folds[(fold + 1) % len(folds)]
    calibration_ids = np.intersect1d(fit_ids, validation_ids)
    inner_known_ids = np.setdiff1d(fit_ids, calibration_ids)

    pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=4080 + fold)
    fit_latent = pca.fit_transform(
        np.sqrt(np.maximum(target[fit_ids], 0.0)).reshape(len(fit_ids), -1)
    )
    clusterer = KMeans(
        n_clusters=MODE_COUNT,
        n_init=20,
        random_state=4080 + fold,
    ).fit(fit_latent)
    cluster_by_id = np.full(len(target), -1, dtype=np.int16)
    cluster_by_id[fit_ids] = clusterer.labels_.astype(np.int16)
    fit_lookup = np.full(len(target), -1, dtype=np.int64)
    fit_lookup[fit_ids] = np.arange(len(fit_ids))

    matrix = graph_matrix(valid_ids, xy)
    valid_lookup = np.full(len(target), -1, dtype=np.int64)
    valid_lookup[valid_ids] = np.arange(len(valid_ids))

    inner_label = cluster_by_id[inner_known_ids]
    inner_direct, inner_graph = propagate_probabilities(
        valid_ids, inner_known_ids, inner_label, xy, matrix
    )
    inner_unknown_ids = np.setdiff1d(valid_ids, inner_known_ids)
    inner_unknown_lookup = np.full(len(target), -1, dtype=np.int64)
    inner_unknown_lookup[inner_unknown_ids] = np.arange(len(inner_unknown_ids))
    calibration_anchor, calibration_donor = query_anchor_and_donors(
        calibration_ids, inner_known_ids, xy, target
    )
    calibration_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    calibration_selection: dict[str, object] = {}
    for name, probability in (
        ("direct", inner_direct),
        ("graph", inner_graph),
    ):
        probability = probability[inner_unknown_lookup[calibration_ids]]
        rank, confidence = select_by_mode(
            probability, calibration_donor, cluster_by_id
        )
        calibration_predictions[name] = (rank, confidence)
        alpha, threshold, delta, grid = select_action(
            calibration_anchor,
            calibration_donor,
            rank,
            confidence,
            target,
            target[calibration_ids],
        )
        calibration_selection[name] = {
            "alpha": alpha,
            "threshold": threshold,
            "delta": delta,
            "grid": grid,
        }

    shuffled_cluster = cluster_by_id.copy()
    rng = np.random.default_rng(40800 + fold)
    shuffled_cluster[fit_ids] = shuffled_cluster[rng.permutation(fit_ids)]
    shuffled_inner_label = shuffled_cluster[inner_known_ids]
    _, shuffled_inner_graph = propagate_probabilities(
        valid_ids, inner_known_ids, shuffled_inner_label, xy, matrix
    )
    shuffled_probability = shuffled_inner_graph[
        inner_unknown_lookup[calibration_ids]
    ]
    shuffled_rank, shuffled_confidence = select_by_mode(
        shuffled_probability, calibration_donor, shuffled_cluster
    )
    shuffled_action = select_action(
        calibration_anchor,
        calibration_donor,
        shuffled_rank,
        shuffled_confidence,
        target,
        target[calibration_ids],
    )

    fit_label = cluster_by_id[fit_ids]
    outer_direct_all, outer_graph_all = propagate_probabilities(
        valid_ids, fit_ids, fit_label, xy, matrix
    )
    outer_unknown_ids = np.setdiff1d(valid_ids, fit_ids)
    outer_lookup = np.full(len(target), -1, dtype=np.int64)
    outer_lookup[outer_unknown_ids] = np.arange(len(outer_unknown_ids))
    outer_anchor, outer_donor = query_anchor_and_donors(
        query_ids, fit_ids, xy, target
    )
    prediction: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    for name, probability_all in (
        ("direct", outer_direct_all),
        ("graph", outer_graph_all),
    ):
        probability = probability_all[outer_lookup[query_ids]]
        rank, confidence = select_by_mode(probability, outer_donor, cluster_by_id)
        selection = calibration_selection[name]
        prediction[name] = render(
            outer_anchor,
            outer_donor,
            rank,
            confidence,
            target,
            selection["alpha"],
            selection["threshold"],
        )
        query_truth_latent = pca.transform(
            np.sqrt(np.maximum(target[query_ids], 0.0)).reshape(len(query_ids), -1)
        )
        truth_cluster = clusterer.predict(query_truth_latent)
        diagnostics[name] = {
            "mode_accuracy": float(
                np.mean(np.argmax(probability, axis=1) == truth_cluster)
            ),
            "confidence": summarize(confidence),
            "active_fraction": float(
                np.mean(confidence >= selection["threshold"])
            ),
        }

    shuffled_outer_label = shuffled_cluster[fit_ids]
    _, shuffled_outer_graph = propagate_probabilities(
        valid_ids, fit_ids, shuffled_outer_label, xy, matrix
    )
    shuffled_probability = shuffled_outer_graph[outer_lookup[query_ids]]
    shuffled_rank, shuffled_confidence = select_by_mode(
        shuffled_probability, outer_donor, shuffled_cluster
    )
    prediction["shuffled"] = render(
        outer_anchor,
        outer_donor,
        shuffled_rank,
        shuffled_confidence,
        target,
        shuffled_action[0],
        shuffled_action[1],
    )

    query_truth_latent = pca.transform(
        np.sqrt(np.maximum(target[query_ids], 0.0)).reshape(len(query_ids), -1)
    )
    truth_cluster = clusterer.predict(query_truth_latent)
    donor_cluster = cluster_by_id[outer_donor]
    oracle_rank = np.zeros(len(query_ids), dtype=np.int64)
    for row in range(len(query_ids)):
        matched = np.flatnonzero(donor_cluster[row] == truth_cluster[row])
        oracle_rank[row] = int(matched[0]) if len(matched) else 0
    oracle_alpha, _, _, _ = select_action(
        calibration_anchor,
        calibration_donor,
        calibration_predictions["graph"][0],
        np.ones(len(calibration_ids), dtype=np.float32),
        target,
        target[calibration_ids],
    )
    prediction["true_cluster"] = render(
        outer_anchor,
        outer_donor,
        oracle_rank,
        np.ones(len(query_ids), dtype=np.float32),
        target,
        oracle_alpha,
        0.0,
    )
    oracle_choice = v406.option_scores(
        outer_anchor, target[outer_donor], target[query_ids]
    ).argmax(axis=1)
    prediction["full_oracle"] = v406.render_choices(
        v406.RouterData(
            query_ids=query_ids,
            donor_ids=outer_donor,
            anchor=outer_anchor,
            scores=np.empty((len(query_ids), 1 + v406.DONOR_K * len(v406.ALPHAS))),
            q_geometry=np.empty((0,)),
            donor_geometry=np.empty((0,)),
            relation_geometry=np.empty((0,)),
            q_full=np.empty((0,)),
            donor_full=np.empty((0,)),
            relation_full=np.empty((0,)),
        ),
        oracle_choice,
        target,
    )

    truth = target[query_ids]
    methods: dict[str, object] = {}
    deltas: dict[str, np.ndarray] = {}
    regimes = regime_of(blocked[query_ids]).astype(str)
    for name in ("direct", "graph", "shuffled", "true_cluster", "full_oracle"):
        result, delta = evaluate(prediction[name], outer_anchor, truth)
        result["by_regime"] = {
            regime: float(np.mean(delta[regimes == regime]))
            for regime in ("clear", "light", "medium", "heavy")
            if np.any(regimes == regime)
        }
        methods[name] = result
        deltas[name] = delta
    row = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "baseline": v402.metric_summary(outer_anchor, truth),
        "methods": methods,
        "calibration": calibration_selection,
        "shuffled_calibration": {
            "alpha": shuffled_action[0],
            "threshold": shuffled_action[1],
            "delta": shuffled_action[2],
        },
        "diagnostics": diagnostics,
        "cluster_coverage": float(
            np.mean(np.any(donor_cluster == truth_cluster[:, None], axis=1))
        ),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "outer_truth_in_deployable_assignment": False,
            "outer_truth_used_for_scoring_and_true_cluster_oracles_only": True,
        },
    }
    arrays = {
        "ids": query_ids,
        "baseline": outer_anchor,
        "truth": truth,
        **prediction,
        **{f"delta_{name}": value for name, value in deltas.items()},
    }
    return row, arrays


def main() -> None:
    for path in (TARGET_CACHE, GEOMETRY_CACHE, v297.COMPONENT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(TARGET_CACHE)
    target = v406.normalize(targets["sh"].astype(np.float32))
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    xy = positions[:, :2]
    blocked = np.load(GEOMETRY_CACHE)["train_blocked"].astype(np.float32)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        targets["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, xy)
    rows: list[dict[str, object]] = []
    parts: list[dict[str, np.ndarray]] = []
    for fold in range(len(folds)):
        row, arrays = fold_experiment(
            fold, folds, valid_ids, xy, target, blocked
        )
        rows.append(row)
        parts.append(arrays)
        print(
            f"v408 fold={fold} direct={row['methods']['direct']['delta']['weighted']:+.5f} "
            f"graph={row['methods']['graph']['delta']['weighted']:+.5f} "
            f"shuffle={row['methods']['shuffled']['delta']['weighted']:+.5f} "
            f"cluster={row['methods']['true_cluster']['delta']['weighted']:+.5f} "
            f"oracle={row['methods']['full_oracle']['delta']['weighted']:+.5f}",
            flush=True,
        )

    aggregate: dict[str, object] = {}
    for name in ("direct", "graph", "shuffled", "true_cluster", "full_oracle"):
        delta = np.asarray(
            [row["methods"][name]["delta"]["weighted"] for row in rows]
        )
        aggregate[name] = {
            "weighted_delta": summarize(delta),
            "positive_folds": int(np.sum(delta > 0.0)),
        }
    graph_delta = float(aggregate["graph"]["weighted_delta"]["mean"])
    control = max(
        float(aggregate[name]["weighted_delta"]["mean"])
        for name in ("direct", "shuffled")
    )
    passed = bool(
        graph_delta >= MIN_ROUTER_DELTA
        and aggregate["graph"]["positive_folds"] >= MIN_FOLD_WINS
        and graph_delta - control >= MIN_CONTROL_MARGIN
    )
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != len(np.unique(ids)) or len(ids) != len(valid_ids):
        raise AssertionError("OOF IDs are not a unique cover")
    names = (
        "baseline",
        "truth",
        "direct",
        "graph",
        "shuffled",
        "true_cluster",
        "full_oracle",
        "delta_direct",
        "delta_graph",
        "delta_shuffled",
        "delta_true_cluster",
        "delta_full_oracle",
    )
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate(
            [
                np.full(len(part["ids"]), fold, dtype=np.int16)
                for fold, part in enumerate(parts)
            ]
        ),
        **{
            name: np.concatenate([part[name] for part in parts])
            for name in names
        },
    )
    report = {
        "method": VERSION,
        "status": (
            "official_oof_passed_ready_for_ai_carrier_deployment"
            if passed
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Propagation mode is a spatially coherent discrete field, so all unknown "
            "UEs should be assigned jointly instead of independently routing donor IDs."
        ),
        "compliance": {
            "uses": [
                "official train H spectra",
                "official train/test coordinates",
                "fit-only KMeans propagation modes and transductive graph",
            ],
            "does_not_use": [
                "test H or hidden truth",
                "v212/Sionna/RT/synthetic targets or online labels",
            ],
        },
        "parameters": {
            "mode_count": MODE_COUNT,
            "pca_dim": PCA_DIM,
            "graph_k": GRAPH_K,
            "graph_mix": GRAPH_MIX,
            "graph_iterations": GRAPH_ITERATIONS,
            "alphas": ALPHAS.tolist(),
            "confidence_thresholds": CONFIDENCE_THRESHOLDS.tolist(),
        },
        "fold_rows": rows,
        "aggregate": aggregate,
        "selection": {
            "minimum_router_delta": MIN_ROUTER_DELTA,
            "minimum_positive_folds": MIN_FOLD_WINS,
            "minimum_control_margin": MIN_CONTROL_MARGIN,
            "observed_control_margin": graph_delta - control,
            "passed": passed,
            "next_action": (
                "neuralize the passed mode field and deploy on the strict v284 carrier"
                if passed
                else "the v405 donor oracle is not observable from spatial mode continuity"
            ),
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
