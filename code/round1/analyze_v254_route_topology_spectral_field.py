#!/usr/bin/env python3
"""v254: official-coordinate route-topology spectral field audit.

The official setup declares 2500 sampled UE positions while the released data
contains 2000 labelled train points and 500 unlabelled test points. This audit
tests whether the joint coordinates reveal road/route topology that improves
official S-H/PDP interpolation over isotropic Euclidean KNN and Delaunay.

Validation points keep their coordinates in the transductive graph but their
channel labels are removed, matching how public test coordinates may be used.
Local replay truth is read at most once, and only when a route method wins the
official five-split comparison by a stable margin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay, cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 254
VERSION = "v254_route_topology_spectral_field"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v254_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v254_route_spectral_targets.npz"
ACTIONS = OUT_DIR / "v254_route_spectral_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

LOCAL_GATE = 0.645604
VAL_COUNT = 320
FOLD_SEEDS = (254, 1254, 2254, 3254, 4254)
TANGENT_K = 16
SEARCH_K = 72
MIN_ROUTE_DELTA = 5e-4


@dataclass(frozen=True)
class Method:
    family: str
    params: tuple[float, ...]

    @property
    def name(self) -> str:
        values = "_".join(f"{value:g}" for value in self.params)
        return f"{self.family}_{values}" if values else self.family


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape(
        (int(np.sum(valid)),) + (1,) * (out.ndim - 1)
    )
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def metrics(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def weighted_bank(
    bank: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    shape = weights.shape + (1,) * (bank.ndim - 1)
    return normalize(
        np.sum(
            bank[indices] * weights.reshape(shape),
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)
    )


def compute_tangents(pos: np.ndarray, k: int = TANGENT_K) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(pos)
    distance, indices = tree.query(pos, k=k + 1)
    tangents = np.zeros((len(pos), 2), dtype=np.float64)
    ratios = np.zeros(len(pos), dtype=np.float64)
    for index in range(len(pos)):
        delta = pos[indices[index, 1:]] - pos[index]
        scale = max(float(np.median(distance[index, 1:])), 1e-6)
        weight = np.exp(-distance[index, 1:] / scale)
        covariance = (delta * weight[:, None]).T @ delta / max(float(weight.sum()), 1e-12)
        value, vector = np.linalg.eigh(covariance)
        tangents[index] = vector[:, -1]
        ratios[index] = value[-1] / max(float(value[0]), 1e-8)
    return tangents, ratios


def choose_seeded_matched_indices(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Match test nearest-neighbour density while producing distinct folds."""
    train_tree = cKDTree(train_pos)
    train_distance = train_tree.query(train_pos, k=2)[0][:, 1]
    test_distance = train_tree.query(test_pos, k=1)[0]
    targets = np.quantile(test_distance, (np.arange(count) + 0.5) / count)
    order = np.argsort(train_distance)
    sorted_distance = train_distance[order]
    used = np.zeros(len(train_pos), dtype=bool)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for target in targets:
        center = int(np.searchsorted(sorted_distance, target))
        lo = max(0, center - 96)
        hi = min(len(order), center + 97)
        candidates = order[lo:hi]
        candidates = candidates[~used[candidates]]
        if len(candidates) == 0:
            candidates = np.flatnonzero(~used)
        delta = np.abs(train_distance[candidates] - target)
        candidate_order = np.argsort(delta)[: min(12, len(candidates))]
        shortlist = candidates[candidate_order]
        shortlist_delta = delta[candidate_order]
        scale = max(float(np.median(shortlist_delta)), 1e-6)
        probability = np.exp(-shortlist_delta / scale)
        probability /= probability.sum()
        selected = int(rng.choice(shortlist, p=probability))
        used[selected] = True
        chosen.append(selected)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def euclidean_predict(
    bank_pos: np.ndarray,
    query_pos: np.ndarray,
    bank_sh: np.ndarray,
    bank_d: np.ndarray,
    k: int,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    distance, indices = cKDTree(bank_pos).query(query_pos, k=k)
    if k == 1:
        distance = distance[:, None]
        indices = indices[:, None]
    weight = 1.0 / np.maximum(distance, 0.05) ** power
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-20)
    return weighted_bank(bank_sh, indices, weight), weighted_bank(bank_d, indices, weight)


def delaunay_predict(
    bank_pos: np.ndarray,
    query_pos: np.ndarray,
    bank_sh: np.ndarray,
    bank_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tri = Delaunay(bank_pos)
    simplex = tri.find_simplex(query_pos)
    indices = np.empty((len(query_pos), 3), dtype=np.int64)
    weights = np.zeros((len(query_pos), 3), dtype=np.float64)
    inside = simplex >= 0
    for index in np.flatnonzero(inside):
        transform = tri.transform[simplex[index]]
        bary = transform[:2] @ (query_pos[index] - transform[2])
        indices[index] = tri.simplices[simplex[index]]
        weights[index] = (bary[0], bary[1], 1.0 - bary.sum())
    if np.any(~inside):
        nearest = cKDTree(bank_pos).query(query_pos[~inside], k=1)[1]
        indices[~inside] = np.repeat(nearest[:, None], 3, axis=1)
        weights[~inside, 0] = 1.0
    return weighted_bank(bank_sh, indices, weights), weighted_bank(bank_d, indices, weights)


def anisotropic_predict(
    all_pos: np.ndarray,
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    tangents: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
    k: int,
    power: float,
    perpendicular_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    search = min(SEARCH_K, len(bank_ids))
    _, local_indices = cKDTree(all_pos[bank_ids]).query(all_pos[query_ids], k=search)
    if search == 1:
        local_indices = local_indices[:, None]
    absolute = bank_ids[local_indices]
    delta = all_pos[absolute] - all_pos[query_ids, None, :]
    tangent = tangents[query_ids]
    along = np.einsum("nkd,nd->nk", delta, tangent)
    across = tangent[:, None, 0] * delta[:, :, 1] - tangent[:, None, 1] * delta[:, :, 0]
    route_distance = np.sqrt(along**2 + (perpendicular_multiplier * across) ** 2)
    order = np.argsort(route_distance, axis=1)[:, :k]
    selected = np.take_along_axis(absolute, order, axis=1)
    selected_distance = np.take_along_axis(route_distance, order, axis=1)
    weight = 1.0 / np.maximum(selected_distance, 0.05) ** power
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-20)
    return weighted_bank(sh, selected, weight), weighted_bank(d, selected, weight)


def bilateral_predict(
    all_pos: np.ndarray,
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    tangents: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
    fallback: tuple[np.ndarray, np.ndarray],
    max_perpendicular: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    search = min(SEARCH_K, len(bank_ids))
    _, local_indices = cKDTree(all_pos[bank_ids]).query(all_pos[query_ids], k=search)
    if search == 1:
        local_indices = local_indices[:, None]
    absolute = bank_ids[local_indices]
    delta = all_pos[absolute] - all_pos[query_ids, None, :]
    tangent = tangents[query_ids]
    along = np.einsum("nkd,nd->nk", delta, tangent)
    across = np.abs(
        tangent[:, None, 0] * delta[:, :, 1]
        - tangent[:, None, 1] * delta[:, :, 0]
    )
    selected = np.zeros((len(query_ids), 2), dtype=np.int64)
    weights = np.zeros((len(query_ids), 2), dtype=np.float64)
    bracketed = np.zeros(len(query_ids), dtype=bool)
    for row in range(len(query_ids)):
        allowed = across[row] <= max_perpendicular
        negative = np.flatnonzero(allowed & (along[row] < -0.2))
        positive = np.flatnonzero(allowed & (along[row] > 0.2))
        if len(negative) == 0 or len(positive) == 0:
            continue
        left = negative[
            np.argmin(np.abs(along[row, negative]) + 2.0 * across[row, negative])
        ]
        right = positive[
            np.argmin(np.abs(along[row, positive]) + 2.0 * across[row, positive])
        ]
        left_distance = -along[row, left]
        right_distance = along[row, right]
        denominator = max(left_distance + right_distance, 1e-8)
        selected[row] = (absolute[row, left], absolute[row, right])
        weights[row] = (right_distance / denominator, left_distance / denominator)
        bracketed[row] = True
    pred_sh = fallback[0].copy()
    pred_d = fallback[1].copy()
    if np.any(bracketed):
        pred_sh[bracketed] = weighted_bank(
            sh, selected[bracketed], weights[bracketed]
        )
        pred_d[bracketed] = weighted_bank(d, selected[bracketed], weights[bracketed])
    return pred_sh, pred_d, bracketed


def build_route_graph(
    pos: np.ndarray,
    tangents: np.ndarray,
    graph_k: int,
    mismatch_penalty: float,
) -> sparse.csr_matrix:
    distance, indices = cKDTree(pos).query(pos, k=graph_k + 1)
    rows = np.repeat(np.arange(len(pos), dtype=np.int64), graph_k)
    cols = indices[:, 1:].reshape(-1).astype(np.int64)
    base = distance[:, 1:].reshape(-1).astype(np.float64)
    direction = (pos[cols] - pos[rows]) / np.maximum(base[:, None], 1e-8)
    mismatch = (
        1.0 - np.abs(np.sum(direction * tangents[rows], axis=1))
        + 1.0 - np.abs(np.sum(direction * tangents[cols], axis=1))
    )
    weight = base * (1.0 + mismatch_penalty * mismatch)
    graph = sparse.coo_matrix((weight, (rows, cols)), shape=(len(pos), len(pos))).tocsr()
    return graph.maximum(graph.T)


def geodesic_predict(
    graph: sparse.csr_matrix,
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
    k: int,
    power: float,
    fallback: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance = dijkstra(graph, directed=False, indices=query_ids, limit=180.0)
    bank_distance = distance[:, bank_ids]
    take = min(k, len(bank_ids))
    local = np.argpartition(bank_distance, take - 1, axis=1)[:, :take]
    selected_distance = np.take_along_axis(bank_distance, local, axis=1)
    connected = np.isfinite(selected_distance).all(axis=1)
    selected = bank_ids[local]
    weight = 1.0 / np.maximum(selected_distance, 0.05) ** power
    weight[~np.isfinite(weight)] = 0.0
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-20)
    pred_sh = weighted_bank(sh, selected, weight)
    pred_d = weighted_bank(d, selected, weight)
    pred_sh[~connected] = fallback[0][~connected]
    pred_d[~connected] = fallback[1][~connected]
    return pred_sh, pred_d, connected


def candidate_methods() -> list[Method]:
    methods: list[Method] = []
    for k in (2, 4, 8, 12, 16, 24):
        for power in (1.0, 1.5, 2.0, 3.0):
            methods.append(Method("euclidean", (k, power)))
    methods.append(Method("delaunay", ()))
    for multiplier in (1.5, 2.0, 3.0, 4.0, 6.0):
        for k in (4, 8, 12, 16):
            for power in (1.0, 2.0):
                methods.append(Method("anisotropic", (k, power, multiplier)))
    for max_perpendicular in (1.5, 2.5, 4.0, 6.0, 9.0):
        methods.append(Method("bilateral", (max_perpendicular,)))
    for graph_k in (6, 10, 16):
        for penalty in (1.0, 3.0, 8.0):
            for k in (4, 8, 16):
                methods.append(Method("geodesic", (graph_k, penalty, k, 2.0)))
    return methods


def main() -> None:
    if not TARGET_CACHE.is_file():
        raise FileNotFoundError(TARGET_CACHE)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    all_pos = np.vstack([train_pos, test_pos])
    test_ids = np.arange(len(train_pos), len(all_pos), dtype=np.int64)
    target_data = np.load(TARGET_CACHE)
    sh = target_data["sh"].astype(np.float32)
    d = target_data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    tangents, tangent_ratio = compute_tangents(all_pos)

    methods = candidate_methods()
    fold_results: dict[str, list[dict[str, float]]] = {method.name: [] for method in methods}
    fold_meta: list[dict[str, object]] = []
    graph_cache: dict[tuple[int, float], sparse.csr_matrix] = {}
    for graph_k in (6, 10, 16):
        for penalty in (1.0, 3.0, 8.0):
            graph_cache[(graph_k, penalty)] = build_route_graph(
                all_pos, tangents, graph_k, penalty
            )

    for fold_index, seed in enumerate(FOLD_SEEDS):
        matched = choose_seeded_matched_indices(train_pos, test_pos, VAL_COUNT, seed)
        val_ids = np.intersect1d(matched, valid_ids)
        fit_ids = np.setdiff1d(valid_ids, val_ids)
        truth_sh = sh[val_ids]
        truth_d = d[val_ids]
        fallback = euclidean_predict(
            train_pos[fit_ids], train_pos[val_ids], sh[fit_ids], d[fit_ids], 8, 2.0
        )
        fold_meta.append(
            {
                "fold": fold_index,
                "seed": seed,
                "fit_count": int(len(fit_ids)),
                "validation_count": int(len(val_ids)),
            }
        )
        for method in methods:
            if method.family == "euclidean":
                k, power = method.params
                pred = euclidean_predict(
                    train_pos[fit_ids],
                    train_pos[val_ids],
                    sh[fit_ids],
                    d[fit_ids],
                    int(k),
                    float(power),
                )
            elif method.family == "delaunay":
                pred = delaunay_predict(
                    train_pos[fit_ids], train_pos[val_ids], sh[fit_ids], d[fit_ids]
                )
            elif method.family == "anisotropic":
                k, power, multiplier = method.params
                pred = anisotropic_predict(
                    all_pos,
                    val_ids,
                    fit_ids,
                    tangents,
                    sh,
                    d,
                    int(k),
                    float(power),
                    float(multiplier),
                )
            elif method.family == "bilateral":
                pred_sh, pred_d, _ = bilateral_predict(
                    all_pos,
                    val_ids,
                    fit_ids,
                    tangents,
                    sh,
                    d,
                    fallback,
                    float(method.params[0]),
                )
                pred = (pred_sh, pred_d)
            elif method.family == "geodesic":
                graph_k, penalty, k, power = method.params
                pred_sh, pred_d, _ = geodesic_predict(
                    graph_cache[(int(graph_k), float(penalty))],
                    val_ids,
                    fit_ids,
                    sh,
                    d,
                    int(k),
                    float(power),
                    fallback,
                )
                pred = (pred_sh, pred_d)
            else:
                raise ValueError(method.family)
            fold_results[method.name].append(metrics(pred[0], pred[1], truth_sh, truth_d))
        print(f"v254 official fold {fold_index + 1}/{len(FOLD_SEEDS)}", flush=True)

    summary: dict[str, dict[str, object]] = {}
    for method in methods:
        rows = fold_results[method.name]
        weighted = np.asarray([row["weighted"] for row in rows], dtype=np.float64)
        summary[method.name] = {
            "family": method.family,
            "params": list(method.params),
            "mean": {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("sh", "d", "weighted")
            },
            "std_weighted": float(np.std(weighted)),
            "folds": rows,
        }

    euclidean_best_name = max(
        (method.name for method in methods if method.family == "euclidean"),
        key=lambda name: float(summary[name]["mean"]["weighted"]),
    )
    spatial_baseline_name = max(
        (euclidean_best_name, "delaunay"),
        key=lambda name: float(summary[name]["mean"]["weighted"]),
    )
    route_best_name = max(
        (
            method.name
            for method in methods
            if method.family in {"anisotropic", "bilateral", "geodesic"}
        ),
        key=lambda name: float(summary[name]["mean"]["weighted"]),
    )
    baseline_folds = np.asarray(
        [row["weighted"] for row in fold_results[spatial_baseline_name]], dtype=np.float64
    )
    route_folds = np.asarray(
        [row["weighted"] for row in fold_results[route_best_name]], dtype=np.float64
    )
    route_delta = float(route_folds.mean() - baseline_folds.mean())
    route_wins = int(np.sum(route_folds > baseline_folds))
    route_selected = bool(route_delta >= MIN_ROUTE_DELTA and route_wins >= 3)

    nearest_distance, _ = cKDTree(train_pos[valid_ids]).query(test_pos, k=8)
    fallback_test = euclidean_predict(
        train_pos[valid_ids], test_pos, sh[valid_ids], d[valid_ids], 8, 2.0
    )
    _, _, bracketed_test = bilateral_predict(
        all_pos,
        test_ids,
        valid_ids,
        tangents,
        sh,
        d,
        fallback_test,
        4.0,
    )

    local_score = None
    diagnostics = None
    selected_method = next(method for method in methods if method.name == route_best_name)
    if route_selected:
        if selected_method.family == "anisotropic":
            k, power, multiplier = selected_method.params
            test_sh, test_d = anisotropic_predict(
                all_pos,
                test_ids,
                valid_ids,
                tangents,
                sh,
                d,
                int(k),
                float(power),
                float(multiplier),
            )
            extra = {}
        elif selected_method.family == "bilateral":
            test_sh, test_d, bracketed = bilateral_predict(
                all_pos,
                test_ids,
                valid_ids,
                tangents,
                sh,
                d,
                fallback_test,
                float(selected_method.params[0]),
            )
            extra = {"bracketed": bracketed}
        elif selected_method.family == "geodesic":
            graph_k, penalty, k, power = selected_method.params
            test_sh, test_d, connected = geodesic_predict(
                graph_cache[(int(graph_k), float(penalty))],
                test_ids,
                valid_ids,
                sh,
                d,
                int(k),
                float(power),
                fallback_test,
            )
            extra = {"connected": connected}
        else:
            raise ValueError(selected_method.family)
        np.savez_compressed(
            TARGETS,
            sh=test_sh,
            d=test_d,
            method=np.asarray(route_best_name),
            **extra,
        )
        diagnostics = v247.project(test_sh, test_d, OUTPUT)
        np.savez_compressed(ACTIONS, **diagnostics)
        local_score = v208.score_prediction(OUTPUT)

    ranking = sorted(
        summary,
        key=lambda name: float(summary[name]["mean"]["weighted"]),
        reverse=True,
    )
    report = {
        "method": VERSION,
        "status": (
            "gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "The 2000 train and 500 test coordinates are a split of one sampled UE route set, "
            "so transductive local tangent or route-geodesic interpolation should outperform "
            "isotropic Euclidean KNN and Delaunay on official spectral holdouts."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H leaked S-H and PDP targets",
                "official train/test coordinates",
                "strict v239 official-data carrier only if route selection passes",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher outputs",
                "local replay truth for method selection",
                "hidden labels",
                "raw RT H",
            ],
            "local_truth_usage": (
                "one final score after official five-fold route selection"
                if route_selected
                else "not read because the route method failed official selection"
            ),
        },
        "parameters": {
            "fold_seeds": list(FOLD_SEEDS),
            "validation_target_count": VAL_COUNT,
            "tangent_k": TANGENT_K,
            "search_k": SEARCH_K,
            "minimum_route_delta": MIN_ROUTE_DELTA,
        },
        "coordinate_diagnostics": {
            "combined_count": int(len(all_pos)),
            "tangent_ratio": summarize(tangent_ratio),
            "test_nearest_train_distance": summarize(nearest_distance[:, 0]),
            "test_eighth_train_distance": summarize(nearest_distance[:, -1]),
            "test_bilateral_bracket_fraction_at_4m": float(np.mean(bracketed_test)),
        },
        "folds": fold_meta,
        "selection": {
            "best_euclidean": euclidean_best_name,
            "spatial_baseline": spatial_baseline_name,
            "best_route": route_best_name,
            "route_delta_mean": route_delta,
            "route_wins": route_wins,
            "route_selected": route_selected,
            "top_20": [
                {"name": name, **summary[name]} for name in ranking[:20]
            ],
        },
        "all_method_summary": summary,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in diagnostics.items()}
            if diagnostics is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(
                    float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
                ),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "targets": PATHS.relative(TARGETS) if TARGETS.is_file() else None,
            "actions": PATHS.relative(ACTIONS) if ACTIONS.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
