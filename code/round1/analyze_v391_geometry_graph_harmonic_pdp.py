#!/usr/bin/env python3
"""Audit official-only geometry graph harmonic interpolation of normalized D."""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v365_multiview_component_pdp as v365
import lifecycle_geometry_features as lgf
from project_paths import PATHS


VERSION_NUMBER = 391
VERSION = "v391_geometry_graph_harmonic_pdp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v391_geometry_graph_harmonic_pdp_oof.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

GRAPH_NEIGHBORS = 24
LOCAL_SCALE_NEIGHBOR = 8
ANCHOR_NEIGHBORS = 16
ANCHOR_DISTANCE_SCALE = 20.0
GEOMETRY_PCA_COMPONENTS = 12
GEOMETRY_STRENGTH = 0.35
RIDGE_FRACTION = 0.05
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

MIN_COSINE_LIFT = 0.005
MIN_GEOMETRY_VS_XY = 0.002
MIN_GEOMETRY_VS_SHUFFLED = 0.002
MIN_EMD_REDUCTION = 0.02
MIN_FOLD_WINS = 4


def normalize_rows(value: np.ndarray) -> np.ndarray:
    return v365.normalize_rows(np.maximum(value, 0.0).astype(np.float32))


def build_graph(
    xy: np.ndarray,
    geometry: np.ndarray | None,
) -> sparse.csr_matrix:
    take = min(GRAPH_NEIGHBORS + 1, len(xy))
    distance, neighbor = cKDTree(xy).query(xy, k=take)
    distance = distance[:, 1:]
    neighbor = neighbor[:, 1:]
    scale_column = min(LOCAL_SCALE_NEIGHBOR - 1, distance.shape[1] - 1)
    local_scale = np.maximum(distance[:, scale_column], 1.0)
    source = np.repeat(np.arange(len(xy)), distance.shape[1])
    target = neighbor.reshape(-1)
    spatial_weight = np.exp(
        -(distance.reshape(-1) ** 2)
        / np.maximum(local_scale[source] * local_scale[target], 1e-8)
    )
    weight = spatial_weight
    if geometry is not None:
        difference = geometry[source] - geometry[target]
        geometry_distance = np.sum(difference**2, axis=1)
        positive = geometry_distance[geometry_distance > 1e-10]
        scale = float(np.median(positive)) if len(positive) else 1.0
        weight = weight * np.exp(
            -GEOMETRY_STRENGTH * geometry_distance / max(scale, 1e-8)
        )
    graph = sparse.coo_matrix(
        (weight.astype(np.float64), (source, target)), shape=(len(xy), len(xy))
    ).tocsr()
    graph = graph.maximum(graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def anchor_prediction(
    xy: np.ndarray,
    truth: np.ndarray,
    fit_rows: np.ndarray,
    query_rows: np.ndarray,
) -> np.ndarray:
    take = min(ANCHOR_NEIGHBORS, len(fit_rows))
    distance, local = cKDTree(xy[fit_rows]).query(xy[query_rows], k=take)
    if take == 1:
        distance = distance[:, None]
        local = local[:, None]
    weight = np.exp(-distance / ANCHOR_DISTANCE_SCALE)
    weight /= np.maximum(np.sum(weight, axis=1, keepdims=True), 1e-20)
    return normalize_rows(
        np.sum(weight[:, :, None] * truth[fit_rows[local]], axis=1)
    )


def harmonic_prediction(
    graph: sparse.csr_matrix,
    xy: np.ndarray,
    truth: np.ndarray,
    fit_rows: np.ndarray,
    unknown_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = anchor_prediction(xy, truth, fit_rows, unknown_rows)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    w_uu = graph[unknown_rows][:, unknown_rows]
    w_ul = graph[unknown_rows][:, fit_rows]
    ridge = RIDGE_FRACTION * max(float(np.median(degree[unknown_rows])), 1e-6)
    system = sparse.diags(degree[unknown_rows] + ridge) - w_uu
    label = np.sqrt(np.maximum(truth[fit_rows], 0.0))
    anchor_sqrt = np.sqrt(np.maximum(anchor, 0.0))
    rhs = w_ul @ label + ridge * anchor_sqrt
    solved = np.asarray(spsolve(system.tocsc(), rhs), dtype=np.float64)
    proposal = normalize_rows(np.maximum(solved, 0.0) ** 2)
    return anchor, proposal


def metric(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    cosine = v365.row_cos(prediction, truth)
    emd = np.mean(
        np.abs(np.cumsum(truth, axis=1) - np.cumsum(prediction, axis=1))
    )
    return {"cosine": float(np.mean(cosine)), "emd": float(emd)}


def select_alpha(
    inner_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHAS:
        scores = []
        emd = []
        for truth, anchor, proposal in inner_rows:
            prediction = normalize_rows(anchor + alpha * (proposal - anchor))
            current = metric(truth, prediction)
            scores.append(current["cosine"])
            emd.append(current["emd"])
        rows.append(
            {
                "alpha": float(alpha),
                "cosine": float(np.mean(scores)),
                "emd": float(np.mean(emd)),
            }
        )
    selected = max(rows, key=lambda row: (row["cosine"], -row["emd"]))
    return float(selected["alpha"]), rows


def mode_inner_rows(
    graph: sparse.csr_matrix,
    xy: np.ndarray,
    truth: np.ndarray,
    fit_rows: np.ndarray,
    outer_rows: np.ndarray,
    folds_local: list[np.ndarray],
    outer_fold: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output = []
    for inner_fold, inner_rows_all in enumerate(folds_local):
        if inner_fold == outer_fold:
            continue
        inner_rows = np.intersect1d(fit_rows, inner_rows_all)
        inner_fit = np.setdiff1d(fit_rows, inner_rows)
        unknown = np.union1d(inner_rows, outer_rows)
        anchor_unknown, proposal_unknown = harmonic_prediction(
            graph, xy, truth, inner_fit, unknown
        )
        lookup = {int(row): index for index, row in enumerate(unknown)}
        selected = np.asarray([lookup[int(row)] for row in inner_rows], dtype=np.int64)
        output.append(
            (truth[inner_rows], anchor_unknown[selected], proposal_unknown[selected])
        )
    return output


def main() -> None:
    for path in (
        TARGET_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
        v238.GEOM_CACHE,
        v297.COMPONENT_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    target = np.load(TARGET_CACHE)
    truth_all = normalize_rows(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    truth = truth_all[valid_ids]
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    xy = train_pos[valid_ids, :2]
    geometry_all = v274.load_geometry()
    all_pos = geometry_all[0]
    scene_cache = np.load(v274.SCENE_CACHE)
    scene_dict = {key: scene_cache[key] for key in scene_cache.files}
    raster = lgf.raster_from_pointcloud_scene(scene_dict)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    point = lgf.point_features(raster, all_pos, bs)[valid_ids]
    ray = point[:, 7:]
    standardized = StandardScaler().fit_transform(ray)
    components = min(GEOMETRY_PCA_COMPONENTS, standardized.shape[1])
    geometry = PCA(n_components=components, random_state=39101).fit_transform(
        standardized
    ).astype(np.float32)
    rng = np.random.default_rng(39102)
    shuffled_geometry = geometry[rng.permutation(len(geometry))]

    graphs = {
        "xy": build_graph(xy, None),
        "shuffled_geometry": build_graph(xy, shuffled_geometry),
        "geometry": build_graph(xy, geometry),
    }
    global_folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    id_to_local = {int(point_id): row for row, point_id in enumerate(valid_ids)}
    folds = [
        np.asarray([id_to_local[int(point_id)] for point_id in fold], dtype=np.int64)
        for fold in global_folds
    ]
    all_rows = np.arange(len(valid_ids), dtype=np.int64)
    fold_reports = []
    oof: dict[str, list[np.ndarray]] = {
        "ids": [], "truth": [], "anchor": [], "xy": [],
        "shuffled_geometry": [], "geometry": [],
    }
    for outer_fold, outer_rows in enumerate(folds):
        fit_rows = np.setdiff1d(all_rows, outer_rows)
        mode_predictions = {}
        mode_reports = {}
        for mode, graph in graphs.items():
            inner = mode_inner_rows(
                graph, xy, truth, fit_rows, outer_rows, folds, outer_fold
            )
            alpha, alpha_rows = select_alpha(inner)
            anchor, proposal = harmonic_prediction(
                graph, xy, truth, fit_rows, outer_rows
            )
            prediction = normalize_rows(anchor + alpha * (proposal - anchor))
            mode_predictions[mode] = prediction
            mode_reports[mode] = {
                "alpha": alpha,
                "selection_rows": alpha_rows,
                "metrics": metric(truth[outer_rows], prediction),
            }
        anchor = anchor_prediction(xy, truth, fit_rows, outer_rows)
        anchor_metric = metric(truth[outer_rows], anchor)
        full = mode_reports["geometry"]["metrics"]
        xy_metric = mode_reports["xy"]["metrics"]
        shuffled = mode_reports["shuffled_geometry"]["metrics"]
        deltas = {
            "geometry_vs_anchor_cosine": float(
                full["cosine"] - anchor_metric["cosine"]
            ),
            "geometry_vs_xy_cosine": float(
                full["cosine"] - xy_metric["cosine"]
            ),
            "geometry_vs_shuffled_cosine": float(
                full["cosine"] - shuffled["cosine"]
            ),
            "geometry_emd_reduction": float(
                (anchor_metric["emd"] - full["emd"])
                / max(anchor_metric["emd"], 1e-20)
            ),
        }
        fold_reports.append(
            {
                "fold": outer_fold,
                "query_count": int(len(outer_rows)),
                "anchor": anchor_metric,
                "modes": mode_reports,
                "deltas": deltas,
                "leakage_assertions": {
                    "outer_truth_in_graph": 0,
                    "outer_truth_in_alpha_selection": 0,
                    "geometry_uses_h": 0,
                    "query_coordinates_in_graph": 1,
                },
            }
        )
        oof["ids"].append(valid_ids[outer_rows])
        oof["truth"].append(truth[outer_rows])
        oof["anchor"].append(anchor)
        for mode in mode_predictions:
            oof[mode].append(mode_predictions[mode])
        print(
            f"v391 fold={outer_fold} alpha={mode_reports['geometry']['alpha']:.2f} "
            f"delta_anchor={deltas['geometry_vs_anchor_cosine']:+.5f} "
            f"delta_xy={deltas['geometry_vs_xy_cosine']:+.5f} "
            f"delta_shuffle={deltas['geometry_vs_shuffled_cosine']:+.5f}",
            flush=True,
        )
    merged = {key: np.concatenate(value) for key, value in oof.items()}
    order = np.argsort(merged["ids"])
    merged = {key: value[order] for key, value in merged.items()}
    np.savez_compressed(OOF_CACHE, **merged)
    mean = {
        key: float(np.mean([row["deltas"][key] for row in fold_reports]))
        for key in fold_reports[0]["deltas"]
    }
    fold_wins = int(
        sum(
            row["deltas"]["geometry_vs_anchor_cosine"] > 0.0
            and row["deltas"]["geometry_vs_xy_cosine"] > 0.0
            and row["deltas"]["geometry_vs_shuffled_cosine"] > 0.0
            for row in fold_reports
        )
    )
    passed = bool(
        mean["geometry_vs_anchor_cosine"] >= MIN_COSINE_LIFT
        and mean["geometry_vs_xy_cosine"] >= MIN_GEOMETRY_VS_XY
        and mean["geometry_vs_shuffled_cosine"] >= MIN_GEOMETRY_VS_SHUFFLED
        and mean["geometry_emd_reduction"] >= MIN_EMD_REDUCTION
        and fold_wins >= MIN_FOLD_WINS
    )
    summary = {**mean, "fold_wins": fold_wins}
    report = {
        "method": VERSION,
        "status": "geometry_graph_passed" if passed else "analysis_only_no_transfer",
        "scope": "official-only transductive geometry graph normalized-D audit",
        "primary_hypothesis": (
            "Point-cloud ray similarity improves harmonic normalized-D interpolation "
            "beyond local anchor, pure XY graph, and shuffled geometry graph."
        ),
        "parameters": {
            "graph_neighbors": GRAPH_NEIGHBORS,
            "local_scale_neighbor": LOCAL_SCALE_NEIGHBOR,
            "anchor_neighbors": ANCHOR_NEIGHBORS,
            "geometry_pca_components": components,
            "geometry_strength": GEOMETRY_STRENGTH,
            "ridge_fraction": RIDGE_FRACTION,
            "alphas": list(ALPHAS),
            "transform": "sqrt-D harmonic extension then square and normalize",
        },
        "fold_rows": fold_reports,
        "summary": summary,
        "decision_rule": {
            "geometry_vs_anchor_cosine": f">={MIN_COSINE_LIFT}",
            "geometry_vs_xy_cosine": f">={MIN_GEOMETRY_VS_XY}",
            "geometry_vs_shuffled_cosine": f">={MIN_GEOMETRY_VS_SHUFFLED}",
            "geometry_emd_reduction": f">={MIN_EMD_REDUCTION}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "passed": passed,
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "oof_cache": PATHS.relative(OOF_CACHE),
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
