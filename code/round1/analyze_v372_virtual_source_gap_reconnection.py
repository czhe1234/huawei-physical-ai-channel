#!/usr/bin/env python3
"""Audit GSTM-style long-path reconnection across artificial spatial gaps."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags
from scipy.sparse.csgraph import connected_components

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v365_multiview_component_pdp as v365
import analyze_v366_local_eikonal_pdp as v366
import analyze_v370_path_track_lifecycle_pdp as v370
from project_paths import PATHS


VERSION_NUMBER = 372
VERSION = "v372_virtual_source_gap_reconnection"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
ROW_CACHE = OUT_DIR / "v372_virtual_source_gap_rows.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

MIN_TRACK_NODES = 12
MIN_SIDE_NODES = 4
MIN_TRACK_SPAN = 15.0
SIDE_FRACTION = 0.40
SOURCE_MARGIN = 500.0
MAX_SOURCE_HEIGHT = 300.0
VIRTUAL_DELAY_SCALE = 0.15
VIRTUAL_GRADIENT_SCALE = 0.45 * v366.MAX_GRADIENT
RANK_DELAY_SCALE = 0.25
RANK_GRADIENT_SCALE = 0.75 * v366.MAX_GRADIENT

MIN_TOP1_VS_AFFINE = 0.05
MIN_TOP1_VS_SHUFFLED = 0.10
MIN_MRR_VS_AFFINE = 0.05
MIN_FOLD_WINS = 4


@dataclass
class VirtualSource:
    sx: float
    sy: float
    height: float
    offset: float
    fit_delay_rmse: float
    fit_gradient_rmse: float


@dataclass
class AffineSurface:
    origin_x: float
    origin_y: float
    intercept: float
    gradient_x: float
    gradient_y: float


def unwrap_delay(delay: np.ndarray) -> np.ndarray:
    reference = float(np.median(delay))
    return reference + v366.signed_circular_delta(delay, reference)


def initial_source(xy: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(gradient, axis=1)
    reliable = norm >= 0.15 * v366.MAX_GRADIENT
    if int(np.sum(reliable)) < 2:
        mean_direction = np.mean(gradient, axis=0)
        direction_norm = max(float(np.linalg.norm(mean_direction)), 1e-8)
        return np.mean(xy, axis=0) - 100.0 * mean_direction / direction_norm
    direction = gradient[reliable] / norm[reliable, None]
    eye = np.eye(2, dtype=np.float64)
    normal = np.zeros((2, 2), dtype=np.float64)
    rhs = np.zeros(2, dtype=np.float64)
    for point, unit in zip(xy[reliable], direction):
        projector = eye - np.outer(unit, unit)
        normal += projector
        rhs += projector @ point
    normal += 1e-5 * eye
    try:
        return np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        mean_direction = np.mean(direction, axis=0)
        direction_norm = max(float(np.linalg.norm(mean_direction)), 1e-8)
        return np.mean(xy, axis=0) - 100.0 * mean_direction / direction_norm


def virtual_prediction(
    model: VirtualSource, xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    delta = xy - np.asarray([model.sx, model.sy], dtype=np.float64)
    distance = np.sqrt(
        np.sum(delta * delta, axis=1) + model.height * model.height
    )
    delay = model.offset + v366.THEORETICAL_GRADIENT * distance
    gradient = (
        v366.THEORETICAL_GRADIENT
        * delta
        / np.maximum(distance[:, None], 1e-8)
    )
    return delay, gradient


def fit_virtual_source(
    xy: np.ndarray,
    delay: np.ndarray,
    gradient: np.ndarray,
    mass: np.ndarray,
    scene_bounds: tuple[float, float, float, float],
) -> VirtualSource | None:
    if len(xy) < MIN_SIDE_NODES:
        return None
    delay_unwrapped = unwrap_delay(delay)
    source = initial_source(xy, gradient)
    x_min, x_max, y_min, y_max = scene_bounds
    source[0] = np.clip(source[0], x_min - SOURCE_MARGIN, x_max + SOURCE_MARGIN)
    source[1] = np.clip(source[1], y_min - SOURCE_MARGIN, y_max + SOURCE_MARGIN)
    weight = np.sqrt(np.maximum(mass, 1e-5))
    weight /= max(float(np.mean(weight)), 1e-8)
    gradient_reliability = np.clip(
        np.linalg.norm(gradient, axis=1) / v366.MAX_GRADIENT, 0.05, 1.0
    )

    lower = np.asarray(
        [x_min - SOURCE_MARGIN, y_min - SOURCE_MARGIN, 0.0, -256.0],
        dtype=np.float64,
    )
    upper = np.asarray(
        [x_max + SOURCE_MARGIN, y_max + SOURCE_MARGIN, MAX_SOURCE_HEIGHT, 256.0],
        dtype=np.float64,
    )

    def residual(parameter: np.ndarray) -> np.ndarray:
        sx, sy, height, offset = parameter
        delta = xy - np.asarray([sx, sy], dtype=np.float64)
        distance = np.sqrt(np.sum(delta * delta, axis=1) + height * height)
        predicted_delay = offset + v366.THEORETICAL_GRADIENT * distance
        predicted_gradient = (
            v366.THEORETICAL_GRADIENT
            * delta
            / np.maximum(distance[:, None], 1e-8)
        )
        delay_residual = (
            weight
            * (predicted_delay - delay_unwrapped)
            / VIRTUAL_DELAY_SCALE
        )
        gradient_residual = (
            weight[:, None]
            * gradient_reliability[:, None]
            * (predicted_gradient - gradient)
            / VIRTUAL_GRADIENT_SCALE
        )
        return np.concatenate([delay_residual, gradient_residual.ravel()])

    best = None
    for height in (0.0, 25.0, 100.0):
        distance = np.sqrt(
            np.sum((xy - source[None, :]) ** 2, axis=1) + height * height
        )
        offset = float(
            np.median(delay_unwrapped - v366.THEORETICAL_GRADIENT * distance)
        )
        initial = np.asarray([source[0], source[1], height, offset])
        initial = np.minimum(np.maximum(initial, lower + 1e-7), upper - 1e-7)
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=300,
        )
        objective = float(np.mean(residual(result.x) ** 2))
        if best is None or objective < best[0]:
            best = (objective, result.x)
    if best is None or not np.all(np.isfinite(best[1])):
        return None
    sx, sy, height, offset = best[1]
    model = VirtualSource(float(sx), float(sy), float(height), float(offset), 0.0, 0.0)
    predicted_delay, predicted_gradient = virtual_prediction(model, xy)
    delay_error = v366.signed_circular_delta(predicted_delay, delay)
    gradient_error = np.linalg.norm(predicted_gradient - gradient, axis=1)
    model.fit_delay_rmse = float(
        np.sqrt(np.sum(weight * delay_error**2) / np.sum(weight))
    )
    model.fit_gradient_rmse = float(
        np.sqrt(np.sum(weight * gradient_error**2) / np.sum(weight))
    )
    return model


def fit_affine_surface(
    xy: np.ndarray,
    delay: np.ndarray,
    mass: np.ndarray,
) -> AffineSurface:
    origin = np.mean(xy, axis=0)
    design = np.column_stack([np.ones(len(xy)), xy - origin[None, :]])
    target = unwrap_delay(delay)
    weight = np.sqrt(np.maximum(mass, 1e-5))
    weighted = design * weight[:, None]
    normal = design.T @ weighted + np.diag([1e-7, 2e-3, 2e-3])
    rhs = design.T @ (weight * target)
    coefficient = np.linalg.solve(normal, rhs)
    gradient = coefficient[1:]
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm > v366.MAX_GRADIENT:
        gradient *= v366.MAX_GRADIENT / gradient_norm
        coefficient[0] = float(
            np.sum(weight * (target - (xy - origin[None, :]) @ gradient))
            / np.sum(weight)
        )
    return AffineSurface(
        float(origin[0]),
        float(origin[1]),
        float(coefficient[0]),
        float(gradient[0]),
        float(gradient[1]),
    )


def affine_prediction(
    model: AffineSurface, xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.asarray([model.gradient_x, model.gradient_y], dtype=np.float64)
    delay = model.intercept + (
        xy - np.asarray([model.origin_x, model.origin_y])
    ) @ gradient
    return delay, np.broadcast_to(gradient, (len(xy), 2)).copy()


def positive_track_components(
    bundle: v370.TrackBundle,
    fit_ids: np.ndarray,
) -> list[np.ndarray]:
    positive = bundle.arrays["binary"] & np.isin(bundle.node_query_id, fit_ids)
    keep = np.flatnonzero(positive)
    if len(keep) == 0:
        return []
    adjacency = bundle.transition
    masked = diags(positive.astype(np.float64)) @ adjacency @ diags(
        positive.astype(np.float64)
    )
    subgraph = masked[keep][:, keep]
    count, label = connected_components(subgraph, directed=False)
    return [keep[label == component] for component in range(count)]


def principal_projection(xy: np.ndarray) -> tuple[np.ndarray, float]:
    centered = xy - np.mean(xy, axis=0)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ right[0]
    span = float(np.max(projection) - np.min(projection))
    return projection, span


def candidate_rank(
    bundle: v370.TrackBundle,
    true_node: int,
    predicted_delay: float,
    predicted_gradient: np.ndarray,
) -> tuple[int, int, float]:
    query_id = int(bundle.node_query_id[true_node])
    candidates = np.flatnonzero(bundle.node_query_id == query_id)
    candidate_delay = bundle.hypothesis.delay.reshape(-1)[
        bundle.hypothesis.valid.reshape(-1)
    ][candidates]
    candidate_gradient = (
        bundle.hypothesis.feature[:, :, :2].reshape(-1, 2)[
            bundle.hypothesis.valid.reshape(-1)
        ][candidates]
        * v366.MAX_GRADIENT
    )
    delay_error = np.abs(
        v366.signed_circular_delta(candidate_delay, predicted_delay)
    )
    gradient_error = np.linalg.norm(
        candidate_gradient - predicted_gradient[None, :], axis=1
    )
    cost = (
        (delay_error / RANK_DELAY_SCALE) ** 2
        + (gradient_error / RANK_GRADIENT_SCALE) ** 2
    )
    order = np.argsort(cost)
    true_local = int(np.flatnonzero(candidates == true_node)[0])
    rank = int(np.flatnonzero(order == true_local)[0]) + 1
    return rank, len(candidates), float(cost[true_local])


def evaluate_direction(
    bundle: v370.TrackBundle,
    fit_nodes: np.ndarray,
    target_nodes: np.ndarray,
    all_xy: np.ndarray,
    scene_bounds: tuple[float, float, float, float],
) -> tuple[dict[str, np.ndarray], VirtualSource] | None:
    node_query = bundle.node_query_id
    flat_delay = bundle.hypothesis.delay.reshape(-1)[
        bundle.hypothesis.valid.reshape(-1)
    ].astype(np.float64)
    flat_gradient = (
        bundle.hypothesis.feature[:, :, :2].reshape(-1, 2)[
            bundle.hypothesis.valid.reshape(-1)
        ].astype(np.float64)
        * v366.MAX_GRADIENT
    )
    mass = bundle.arrays["mass"].astype(np.float64)
    model = fit_virtual_source(
        all_xy[node_query[fit_nodes]],
        flat_delay[fit_nodes],
        flat_gradient[fit_nodes],
        mass[fit_nodes],
        scene_bounds,
    )
    if model is None:
        return None
    affine = fit_affine_surface(
        all_xy[node_query[fit_nodes]],
        flat_delay[fit_nodes],
        mass[fit_nodes],
    )
    target_xy = all_xy[node_query[target_nodes]]
    virtual_delay, virtual_gradient = virtual_prediction(model, target_xy)
    affine_delay, affine_gradient = affine_prediction(affine, target_xy)
    virtual_rank = []
    affine_rank = []
    candidate_count = []
    virtual_cost = []
    affine_cost = []
    for row, node in enumerate(target_nodes):
        rank, count, cost = candidate_rank(
            bundle, int(node), float(virtual_delay[row]), virtual_gradient[row]
        )
        virtual_rank.append(rank)
        candidate_count.append(count)
        virtual_cost.append(cost)
        rank, _, cost = candidate_rank(
            bundle, int(node), float(affine_delay[row]), affine_gradient[row]
        )
        affine_rank.append(rank)
        affine_cost.append(cost)
    return (
        {
            "node": target_nodes.astype(np.int64),
            "virtual_rank": np.asarray(virtual_rank, dtype=np.int16),
            "affine_rank": np.asarray(affine_rank, dtype=np.int16),
            "candidate_count": np.asarray(candidate_count, dtype=np.int16),
            "virtual_cost": np.asarray(virtual_cost, dtype=np.float32),
            "affine_cost": np.asarray(affine_cost, dtype=np.float32),
        },
        model,
    )


def summarize_rank(rank: np.ndarray) -> dict[str, float]:
    return {
        "top1": float(np.mean(rank <= 1)),
        "top3": float(np.mean(rank <= 3)),
        "mrr": float(np.mean(1.0 / rank)),
        "median_rank": float(np.median(rank)),
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
    x_min = float(np.min(train_pos[:, 0]))
    x_max = float(np.max(train_pos[:, 0]))
    y_min = float(np.min(train_pos[:, 1]))
    y_max = float(np.max(train_pos[:, 1]))
    scene_bounds = (x_min, x_max, y_min, y_max)

    fold_rows: list[dict[str, object]] = []
    cache_rows: dict[str, list[np.ndarray]] = {
        "fold": [],
        "track": [],
        "direction": [],
        "node": [],
        "virtual_rank": [],
        "affine_rank": [],
        "candidate_count": [],
        "virtual_cost": [],
        "affine_cost": [],
    }
    fold_models: list[list[VirtualSource]] = []
    fold_bundles: list[v370.TrackBundle] = []
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        bundle = v370.prepare_bundle(valid_ids, fit_ids, geometry, node_raw, component)
        fold_bundles.append(bundle)
        components = positive_track_components(bundle, fit_ids)
        accepted = []
        models: list[VirtualSource] = []
        duplicate_query_components = 0
        for track, nodes in enumerate(components):
            if len(nodes) < MIN_TRACK_NODES:
                continue
            query_ids = bundle.node_query_id[nodes]
            if len(np.unique(query_ids)) != len(query_ids):
                duplicate_query_components += 1
                continue
            xy = all_pos[query_ids, :2]
            projection, span = principal_projection(xy)
            if span < MIN_TRACK_SPAN:
                continue
            low = float(np.quantile(projection, SIDE_FRACTION))
            high = float(np.quantile(projection, 1.0 - SIDE_FRACTION))
            left = nodes[projection <= low]
            right = nodes[projection >= high]
            if len(left) < MIN_SIDE_NODES or len(right) < MIN_SIDE_NODES:
                continue
            direction_rows = []
            for direction, (fit_nodes, target_nodes) in enumerate(
                ((left, right), (right, left))
            ):
                evaluated = evaluate_direction(
                    bundle,
                    fit_nodes,
                    target_nodes,
                    all_pos[:, :2],
                    scene_bounds,
                )
                if evaluated is None:
                    continue
                values, model = evaluated
                models.append(model)
                direction_rows.append(values)
            if len(direction_rows) == 2:
                accepted.append((track, span, direction_rows))
                for direction, values in enumerate(direction_rows):
                    count = len(values["node"])
                    cache_rows["fold"].append(
                        np.full(count, fold, dtype=np.int8)
                    )
                    cache_rows["track"].append(
                        np.full(count, track, dtype=np.int32)
                    )
                    cache_rows["direction"].append(
                        np.full(count, direction, dtype=np.int8)
                    )
                    for key in (
                        "node",
                        "virtual_rank",
                        "affine_rank",
                        "candidate_count",
                        "virtual_cost",
                        "affine_cost",
                    ):
                        cache_rows[key].append(values[key])

        fold_models.append(models)
        if not accepted:
            raise RuntimeError(f"fold {fold} has no accepted gap tracks")
        virtual_rank = np.concatenate(
            [row["virtual_rank"] for _, _, pair in accepted for row in pair]
        )
        affine_rank = np.concatenate(
            [row["affine_rank"] for _, _, pair in accepted for row in pair]
        )
        candidate_count = np.concatenate(
            [row["candidate_count"] for _, _, pair in accepted for row in pair]
        )
        virtual_cost = np.concatenate(
            [row["virtual_cost"] for _, _, pair in accepted for row in pair]
        )
        affine_cost = np.concatenate(
            [row["affine_cost"] for _, _, pair in accepted for row in pair]
        )
        virtual_summary = summarize_rank(virtual_rank)
        affine_summary = summarize_rank(affine_rank)
        fold_rows.append(
            {
                "fold": fold,
                "all_positive_components": int(len(components)),
                "eligible_gap_tracks": int(len(accepted)),
                "duplicate_query_components_rejected": int(duplicate_query_components),
                "target_candidate_rows": int(len(virtual_rank)),
                "candidate_count_mean": float(np.mean(candidate_count)),
                "track_span_mean_m": float(np.mean([row[1] for row in accepted])),
                "virtual_source": virtual_summary,
                "affine": affine_summary,
                "virtual_vs_affine": {
                    "top1": float(virtual_summary["top1"] - affine_summary["top1"]),
                    "top3": float(virtual_summary["top3"] - affine_summary["top3"]),
                    "mrr": float(virtual_summary["mrr"] - affine_summary["mrr"]),
                },
                "true_candidate_cost": {
                    "virtual_mean": float(np.mean(virtual_cost)),
                    "affine_mean": float(np.mean(affine_cost)),
                    "virtual_median": float(np.median(virtual_cost)),
                    "affine_median": float(np.median(affine_cost)),
                },
            }
        )
        print(
            f"v372 fold={fold} tracks={len(accepted)} rows={len(virtual_rank)} "
            f"top1 virtual={virtual_summary['top1']:.4f} "
            f"affine={affine_summary['top1']:.4f} "
            f"MRR virtual={virtual_summary['mrr']:.4f} "
            f"affine={affine_summary['mrr']:.4f}",
            flush=True,
        )

    merged_cache = {
        key: np.concatenate(value) if value else np.empty(0)
        for key, value in cache_rows.items()
    }
    np.savez_compressed(ROW_CACHE, **merged_cache)
    virtual_rank = merged_cache["virtual_rank"]
    affine_rank = merged_cache["affine_rank"]
    virtual_summary = summarize_rank(virtual_rank)
    affine_summary = summarize_rank(affine_rank)

    rng = np.random.default_rng(372_000)
    shuffled_rank_rows = []
    # Preserve each fold's fitted model distribution but break family identity.
    for fold, models in enumerate(fold_models):
        keep = merged_cache["fold"] == fold
        node = merged_cache["node"][keep]
        if not models or len(node) == 0:
            continue
        shuffled_models = rng.integers(0, len(models), size=len(node))
        # A wrong-family model is evaluated at the same query candidate set.
        bundle = fold_bundles[fold]
        for true_node, model_id in zip(node, shuffled_models):
            model = models[int(model_id)]
            xy = all_pos[[bundle.node_query_id[int(true_node)]], :2]
            predicted_delay, predicted_gradient = virtual_prediction(model, xy)
            rank, _, _ = candidate_rank(
                bundle,
                int(true_node),
                float(predicted_delay[0]),
                predicted_gradient[0],
            )
            shuffled_rank_rows.append(rank)
    shuffled_rank = np.asarray(shuffled_rank_rows, dtype=np.int16)
    shuffled_summary = summarize_rank(shuffled_rank)
    fold_wins = int(
        sum(
            row["virtual_vs_affine"]["top1"] > 0.0
            and row["virtual_vs_affine"]["mrr"] > 0.0
            for row in fold_rows
        )
    )
    top1_vs_affine = float(virtual_summary["top1"] - affine_summary["top1"])
    top1_vs_shuffled = float(virtual_summary["top1"] - shuffled_summary["top1"])
    mrr_vs_affine = float(virtual_summary["mrr"] - affine_summary["mrr"])
    passed = bool(
        top1_vs_affine >= MIN_TOP1_VS_AFFINE
        and top1_vs_shuffled >= MIN_TOP1_VS_SHUFFLED
        and mrr_vs_affine >= MIN_MRR_VS_AFFINE
        and fold_wins >= MIN_FOLD_WINS
    )
    summary = {
        "virtual_source": virtual_summary,
        "affine": affine_summary,
        "shuffled_family": shuffled_summary,
        "virtual_vs_affine_top1": top1_vs_affine,
        "virtual_vs_shuffled_top1": top1_vs_shuffled,
        "virtual_vs_affine_mrr": mrr_vs_affine,
        "fold_wins": fold_wins,
        "target_candidate_rows": int(len(virtual_rank)),
    }
    report = {
        "method": VERSION,
        "status": (
            "virtual_source_gap_reconnection_passed"
            if passed
            else "analysis_only_no_candidate"
        ),
        "scope": "fit-only long-family mechanism audit; no PDP target, full H, test prediction, or ZIP",
        "primary_hypothesis": (
            "A GSTM-style virtual last-hop source surface reconnects the same MPC "
            "after a spatial visibility gap more accurately than a local affine surface."
        ),
        "compliance": {
            "uses": [
                "official train H continuous path labels",
                "official train planar coordinates",
                "v370 fit-only short path tracks and v366 eikonal gradients",
            ],
            "does_not_use": [
                "outer labels in track construction",
                "Sionna/synthetic labels, v212 teacher, local/hidden/online truth",
                "PDP projection, full-H generation, test target, or submission packaging",
            ],
        },
        "parameters": {
            "minimum_track_nodes": MIN_TRACK_NODES,
            "minimum_side_nodes": MIN_SIDE_NODES,
            "minimum_track_span_m": MIN_TRACK_SPAN,
            "artificial_gap": "middle 20% along each track's principal spatial axis",
            "virtual_surface": "offset + theoretical_delay_gradient * 3D distance to effective source",
            "virtual_fit": "mass-weighted robust delay plus eikonal-gradient residual",
            "control": "local affine delay surface and wrong-family virtual source",
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "virtual_vs_affine_top1": f">={MIN_TOP1_VS_AFFINE}",
            "virtual_vs_shuffled_top1": f">={MIN_TOP1_VS_SHUFFLED}",
            "virtual_vs_affine_mrr": f">={MIN_MRR_VS_AFFINE}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "passed": passed,
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "row_cache": PATHS.relative(ROW_CACHE),
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
