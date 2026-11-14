#!/usr/bin/env python3
"""Audit whether the v391 geometry graph complements strict v284 OOF D."""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import analyze_v254_route_topology_spectral_field as v254
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v265_scatterer_delay_angle_oracle as v265
import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
import analyze_v283_joint_geometry_path_renderer as v283
import analyze_v284_planar_geometry_residual_field as v284
import analyze_v391_geometry_graph_harmonic_pdp as v391
import lifecycle_geometry_features as lgf
from project_paths import PATHS


VERSION_NUMBER = 392
VERSION = "v392_v284_geometry_graph_complement"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v392_v284_geometry_graph_complement_oof.npz"
V284_REPORT = PATHS.version_report_dir(284) / "v284_planar_geometry_residual_field.json"

ABSOLUTE_STRENGTHS = (0.10, 0.20, 0.30, 0.50)
INCREMENT_STRENGTHS = (0.25, 0.50, 1.00, 2.00)
MIN_COSINE_LIFT = 5e-4
MIN_VS_XY = 2e-4
MIN_VS_SHUFFLED = 2e-4
MIN_FOLD_WINS = 4


def normalize(value: np.ndarray) -> np.ndarray:
    return v391.normalize_rows(value.astype(np.float32))


def cosine_rows(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return v279.row_cos(truth, prediction).astype(np.float64)


def emd_rows(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.mean(
        np.abs(np.cumsum(truth, axis=1) - np.cumsum(prediction, axis=1)),
        axis=1,
    ).astype(np.float64)


def action_prediction(
    base: np.ndarray,
    anchor: np.ndarray,
    xy: np.ndarray,
    mode: np.ndarray,
    family: str,
    strength: float,
) -> np.ndarray:
    if family == "base":
        return base
    if family == "absolute":
        return normalize(base + strength * (mode - base))
    if family == "increment":
        return normalize(base + strength * (mode - xy))
    raise ValueError(family)


def action_grid(mode_name: str) -> list[tuple[str, str, float]]:
    rows = [(f"{mode_name}:base", "base", 0.0)]
    rows.extend(
        (f"{mode_name}:absolute:{value:.2f}", "absolute", value)
        for value in ABSOLUTE_STRENGTHS
    )
    if mode_name != "xy":
        rows.extend(
            (f"{mode_name}:increment:{value:.2f}", "increment", value)
            for value in INCREMENT_STRENGTHS
        )
    return rows


def score_action(
    fold_data: list[dict[str, np.ndarray]],
    fold_ids: list[int],
    mode_name: str,
    family: str,
    strength: float,
) -> tuple[float, float]:
    cosine_parts = []
    emd_parts = []
    for fold in fold_ids:
        data = fold_data[fold]
        prediction = action_prediction(
            data["base"], data["anchor"], data["xy"], data[mode_name],
            family, strength,
        )
        cosine_parts.append(cosine_rows(data["truth"], prediction))
        emd_parts.append(emd_rows(data["truth"], prediction))
    return (
        float(np.mean(np.concatenate(cosine_parts))),
        float(np.mean(np.concatenate(emd_parts))),
    )


def select_action(
    fold_data: list[dict[str, np.ndarray]],
    fit_folds: list[int],
    mode_name: str,
) -> tuple[tuple[str, str, float], list[dict[str, object]]]:
    rows = []
    actions = action_grid(mode_name)
    for name, family, strength in actions:
        cosine, emd = score_action(
            fold_data, fit_folds, mode_name, family, strength
        )
        rows.append(
            {
                "name": name,
                "family": family,
                "strength": strength,
                "cosine": cosine,
                "emd": emd,
            }
        )
    selected = max(rows, key=lambda row: (row["cosine"], -row["emd"]))
    action = (
        str(selected["name"]),
        str(selected["family"]),
        float(selected["strength"]),
    )
    return action, rows


def load_graph_inputs(
    valid_ids: np.ndarray,
    train_pos: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    geometry_all = v274.load_geometry()
    all_pos = geometry_all[0]
    scene_cache = np.load(v274.SCENE_CACHE)
    scene_dict = {key: scene_cache[key] for key in scene_cache.files}
    raster = lgf.raster_from_pointcloud_scene(scene_dict)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    point = lgf.point_features(raster, all_pos, bs)[valid_ids]
    ray = point[:, 7:]
    standardized = StandardScaler().fit_transform(ray)
    components = min(v391.GEOMETRY_PCA_COMPONENTS, standardized.shape[1])
    geometry = PCA(
        n_components=components, random_state=39101
    ).fit_transform(standardized).astype(np.float32)
    return geometry, {"components": components, "ray_dimension": ray.shape[1]}


def reconstruct_v284_oof() -> tuple[
    list[dict[str, np.ndarray]], dict[str, object]
]:
    for path in (
        v283.TARGET_CACHE,
        v283.TRAIN_CANDIDATES,
        v283.TEST_CANDIDATES,
        v283.SCENE_CACHE,
        v283.V265_REPORT,
        V284_REPORT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    v284.configure_runner()
    target_data = np.load(v283.TARGET_CACHE)
    truth_sh = v279.normalize(target_data["sh"].astype(np.float32))
    truth_d = v279.normalize(target_data["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    patches = np.load(v283.SCENE_CACHE)["wall_patches"].astype(np.float32)
    train_candidates = dict(np.load(v283.TRAIN_CANDIDATES))
    test_candidates = dict(np.load(v283.TEST_CANDIDATES))
    train_node, test_node = v257.build_node_features(
        train_pos[:, :2], test_pos[:, :2]
    )
    train_raw = v264.build_features(
        train_pos, train_candidates, patches, train_node
    )
    test_raw = v264.build_features(test_pos, test_candidates, patches, test_node)
    with v283.V265_REPORT.open("r", encoding="utf-8") as handle:
        best_spec = json.load(handle)["selection"]["best"]
    train_s, train_h = v283.centers(
        train_pos, train_candidates, train_raw, best_spec
    )

    historical = json.loads(V284_REPORT.read_text(encoding="utf-8"))
    selected_epochs = historical["official_selection"]["selected_epochs"]
    residual_alpha = float(historical["official_selection"]["residual_alpha"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v392 reconstruct device={device} valid={len(valid_ids)}", flush=True)

    geometry, geometry_meta = load_graph_inputs(valid_ids, train_pos)
    xy_valid = train_pos[valid_ids, :2]
    rng = np.random.default_rng(39102)
    shuffled_geometry = geometry[rng.permutation(len(geometry))]
    graphs = {
        "xy": v391.build_graph(xy_valid, None),
        "shuffled": v391.build_graph(xy_valid, shuffled_geometry),
        "geometry": v391.build_graph(xy_valid, geometry),
    }
    id_to_local = {int(point_id): row for row, point_id in enumerate(valid_ids)}

    fold_data: list[dict[str, np.ndarray]] = []
    fold_meta = []
    for fold, (seed, selected_epoch) in enumerate(
        zip(v283.OUTER_SEEDS, selected_epochs)
    ):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], v283.OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        outer_fit_raw = train_raw.take(outer_fit)
        outer_feature, outer_full, _outer_test, _ = v264.standardize_features(
            outer_fit_raw, train_raw, test_raw
        )
        outer_val_feature = outer_full.take(outer_ids)
        outer_fit_anchor = v283.knn_anchor(
            outer_fit, outer_fit, train_pos, truth_sh
        )
        outer_val_anchor = v283.knn_anchor(
            outer_ids, outer_fit, train_pos, truth_sh
        )
        if int(selected_epoch) == 0:
            model_sh = outer_val_anchor.copy()
        else:
            model, _training = v283.train_model(
                outer_feature,
                train_s[outer_fit],
                train_h[outer_fit],
                outer_fit_anchor,
                truth_sh[outer_fit],
                truth_d[outer_fit],
                int(seed) + 100_000,
                int(selected_epoch),
                device,
                len(patches),
                None,
            )
            model_sh, _gates = v283.predict(
                model,
                outer_val_feature,
                train_s[outer_ids],
                train_h[outer_ids],
                outer_val_anchor,
                device,
            )
        blended_sh = v279.normalize(
            outer_val_anchor + residual_alpha * (model_sh - outer_val_anchor)
        )
        base_d = v279.derive_d_numpy(blended_sh)

        query_local = np.asarray(
            [id_to_local[int(point_id)] for point_id in outer_ids], dtype=np.int64
        )
        fit_local = np.setdiff1d(
            np.arange(len(valid_ids), dtype=np.int64), query_local
        )
        graph_prediction = {}
        graph_anchor = None
        for mode, graph in graphs.items():
            current_anchor, current_prediction = v391.harmonic_prediction(
                graph, xy_valid, truth_d[valid_ids], fit_local, query_local
            )
            if graph_anchor is None:
                graph_anchor = current_anchor
            graph_prediction[mode] = current_prediction
        assert graph_anchor is not None
        data = {
            "ids": outer_ids,
            "truth": truth_d[outer_ids],
            "base": base_d,
            "anchor": graph_anchor,
            "xy": graph_prediction["xy"],
            "shuffled": graph_prediction["shuffled"],
            "geometry": graph_prediction["geometry"],
        }
        fold_data.append(data)
        base_cosine = float(np.mean(cosine_rows(data["truth"], data["base"])))
        fold_meta.append(
            {
                "fold": fold,
                "seed": int(seed),
                "query_count": int(len(outer_ids)),
                "selected_epoch": int(selected_epoch),
                "reconstructed_v284_d_cosine": base_cosine,
                "historical_v284_d_cosine": float(
                    historical["outer_folds"][fold]["blended"]["d"]
                ),
            }
        )
        print(
            f"v392 fold={fold} epoch={selected_epoch} "
            f"v284_D={base_cosine:.6f}",
            flush=True,
        )
    metadata = {
        "valid_ids": valid_ids,
        "train_pos": train_pos,
        "geometry": geometry_meta,
        "folds": fold_meta,
    }
    return fold_data, metadata


def main() -> None:
    fold_data, metadata = reconstruct_v284_oof()
    nested_rows = []
    cache: dict[str, list[np.ndarray]] = {
        key: [] for key in ("fold", "ids", "truth", "base", "anchor", "xy", "shuffled", "geometry")
    }
    for held_fold in range(len(fold_data)):
        fit_folds = [fold for fold in range(len(fold_data)) if fold != held_fold]
        selected = {}
        selection_rows = {}
        predictions = {}
        for mode_name in ("geometry", "xy", "shuffled"):
            action, rows = select_action(fold_data, fit_folds, mode_name)
            selected[mode_name] = action
            selection_rows[mode_name] = rows
            _name, family, strength = action
            data = fold_data[held_fold]
            predictions[mode_name] = action_prediction(
                data["base"], data["anchor"], data["xy"], data[mode_name],
                family, strength,
            )
        data = fold_data[held_fold]
        base_cosine = float(np.mean(cosine_rows(data["truth"], data["base"])))
        metrics = {}
        for mode_name, prediction in predictions.items():
            metrics[mode_name] = {
                "cosine": float(np.mean(cosine_rows(data["truth"], prediction))),
                "emd": float(np.mean(emd_rows(data["truth"], prediction))),
            }
        delta = {
            "geometry_vs_v284": metrics["geometry"]["cosine"] - base_cosine,
            "geometry_vs_xy": metrics["geometry"]["cosine"] - metrics["xy"]["cosine"],
            "geometry_vs_shuffled": metrics["geometry"]["cosine"] - metrics["shuffled"]["cosine"],
        }
        nested_rows.append(
            {
                "held_fold": held_fold,
                "query_count": int(len(data["ids"])),
                "v284": {
                    "cosine": base_cosine,
                    "emd": float(np.mean(emd_rows(data["truth"], data["base"]))),
                },
                "selected_actions": {
                    key: {"name": value[0], "family": value[1], "strength": value[2]}
                    for key, value in selected.items()
                },
                "selection_rows": selection_rows,
                "metrics": metrics,
                "deltas": delta,
            }
        )
        for key in cache:
            if key == "fold":
                cache[key].append(np.full(len(data["ids"]), held_fold, dtype=np.int8))
            else:
                cache[key].append(data[key])
        print(
            f"v392 nested fold={held_fold} action={selected['geometry'][0]} "
            f"delta={delta['geometry_vs_v284']:+.6f} "
            f"vs_xy={delta['geometry_vs_xy']:+.6f} "
            f"vs_shuffle={delta['geometry_vs_shuffled']:+.6f}",
            flush=True,
        )

    merged = {key: np.concatenate(value) for key, value in cache.items()}
    np.savez_compressed(OOF_CACHE, **merged)
    summary = {
        key: float(np.mean([row["deltas"][key] for row in nested_rows]))
        for key in nested_rows[0]["deltas"]
    }
    summary["fold_wins"] = int(
        sum(row["deltas"]["geometry_vs_v284"] > 0 for row in nested_rows)
    )
    summary["reconstruction_max_abs_cosine_error"] = float(
        max(
            abs(
                row["reconstructed_v284_d_cosine"]
                - row["historical_v284_d_cosine"]
            )
            for row in metadata["folds"]
        )
    )
    passed = bool(
        summary["geometry_vs_v284"] >= MIN_COSINE_LIFT
        and summary["geometry_vs_xy"] >= MIN_VS_XY
        and summary["geometry_vs_shuffled"] >= MIN_VS_SHUFFLED
        and summary["fold_wins"] >= MIN_FOLD_WINS
        and summary["reconstruction_max_abs_cosine_error"] <= 2e-5
    )
    report = {
        "method": VERSION,
        "status": "complement_passed" if passed else "analysis_only_no_transfer",
        "primary_hypothesis": (
            "The causal point-cloud graph direction from v391 corrects normalized-D errors "
            "that remain after the stronger v284 planar geometry residual model."
        ),
        "scope": "official-only normalized global-D complement audit; no test target or ZIP",
        "parameters": {
            "v284_residual_alpha": 0.5,
            "absolute_strengths": list(ABSOLUTE_STRENGTHS),
            "increment_strengths": list(INCREMENT_STRENGTHS),
            "selection": "leave-one-v284-outer-fold-out action-family and strength selection",
            "v391_graph_parameters": {
                "neighbors": v391.GRAPH_NEIGHBORS,
                "local_scale_neighbor": v391.LOCAL_SCALE_NEIGHBOR,
                "geometry_strength": v391.GEOMETRY_STRENGTH,
                "ridge_fraction": v391.RIDGE_FRACTION,
            },
        },
        "reconstruction": metadata["folds"],
        "nested_rows": nested_rows,
        "summary": summary,
        "decision_rule": {
            "geometry_vs_v284": f">={MIN_COSINE_LIFT}",
            "geometry_vs_xy": f">={MIN_VS_XY}",
            "geometry_vs_shuffled": f">={MIN_VS_SHUFFLED}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "reconstruction_max_abs_cosine_error": "<=2e-5",
            "passed": passed,
        },
        "compliance_boundary": {
            "uses": [
                "official train H normalized D labels",
                "official train/test coordinates for test-distance-matched folds",
                "official point-cloud ray geometry",
                "historical v284 architecture and frozen selected epochs",
            ],
            "does_not_use": [
                "v212 outputs or labels",
                "synthetic channel labels",
                "hidden or online truth",
                "test D targets",
            ],
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
