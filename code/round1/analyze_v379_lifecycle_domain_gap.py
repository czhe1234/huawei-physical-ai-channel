#!/usr/bin/env python3
"""Measure feature-group domain gap between v374 and official lifecycle pairs."""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v370_path_track_lifecycle_pdp as v370
import analyze_v373_long_family_visibility as v373
import analyze_v376_frozen_lifecycle_transfer as v376
import lifecycle_geometry_features as lgf
import train_v375_synthetic_lifecycle_pretraining as v375
from project_paths import PATHS


VERSION_NUMBER = 379
VERSION = "v379_lifecycle_domain_gap"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OFFICIAL_CACHE = OUT_DIR / "v379_official_lifecycle_features.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
SYNTHETIC_CACHE = PATHS.cache_dir(375) / "v375_synthetic_lifecycle_examples.npz"

MAX_DOMAIN_ROWS_PER_SOURCE = 60000
DOMAIN_TREES = 300


def feature_groups() -> dict[str, np.ndarray]:
    dimensions = lgf.feature_dimensions()
    point_end = dimensions["point"]
    path_end = point_end + dimensions["path"]
    support_end = path_end + dimensions["support"]
    total = dimensions["total"]
    return {
        "point": np.arange(0, point_end),
        "path": np.arange(point_end, path_end),
        "support": np.arange(path_end, support_end),
        "terminal_ray": np.arange(support_end, total),
        "point_terminal_geometry": np.r_[
            np.arange(0, point_end), np.arange(support_end, total)
        ],
        "path_support": np.arange(point_end, support_end),
        "all": np.arange(total),
    }


def build_official_features() -> dict[str, np.ndarray]:
    if OFFICIAL_CACHE.is_file():
        cached = np.load(OFFICIAL_CACHE)
        return {key: cached[key] for key in cached.files}
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
    scene_cache = np.load(v274.SCENE_CACHE)
    scene_dict = {key: scene_cache[key] for key in scene_cache.files}
    raster = lgf.raster_from_pointcloud_scene(scene_dict)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    point_feature = lgf.point_features(raster, all_pos, bs)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    rows: dict[str, list[np.ndarray]] = {
        "feature": [],
        "binary": [],
        "mass": [],
        "identity": [],
        "fold": [],
        "query_id": [],
    }
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        bundle = v370.prepare_bundle(valid_ids, fit_ids, geometry, node_raw, component)
        families, _ = v373.discover_families(bundle, fit_ids, all_pos[:, :2])
        exposure = v373.build_exposures(
            families, bundle, outer_ids, all_pos[:, :2]
        )
        feature = v376.exposure_features(
            exposure=exposure,
            families=families,
            bundle=bundle,
            all_pos=all_pos,
            point_feature=point_feature,
            raster=raster,
            bs=bs,
        )
        rows["feature"].append(feature)
        rows["binary"].append(exposure.binary.astype(np.int8))
        rows["mass"].append(exposure.mass.astype(np.float32))
        rows["identity"].append(exposure.identity.astype(np.float32))
        rows["fold"].append(np.full(len(feature), fold, dtype=np.int8))
        rows["query_id"].append(exposure.query_id.astype(np.int64))
        print(f"v379 official fold={fold} exposures={len(feature)}", flush=True)
    output = {key: np.concatenate(value) for key, value in rows.items()}
    np.savez_compressed(OFFICIAL_CACHE, **output)
    return output


def balanced_domain_rows(
    synthetic_rows: np.ndarray,
    official_rows: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    take = min(
        len(synthetic_rows),
        len(official_rows),
        MAX_DOMAIN_ROWS_PER_SOURCE,
    )
    synthetic_take = rng.choice(synthetic_rows, size=take, replace=False)
    official_take = rng.choice(official_rows, size=take, replace=False)
    return synthetic_take, official_take


def domain_split(
    synthetic_scene: np.ndarray,
    official_fold: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "train": (
            np.flatnonzero(np.isin(synthetic_scene, np.arange(18))),
            np.flatnonzero(np.isin(official_fold, (0, 1, 2))),
        ),
        "validation": (
            np.flatnonzero(np.isin(synthetic_scene, (18, 19, 20))),
            np.flatnonzero(official_fold == 3),
        ),
        "test": (
            np.flatnonzero(np.isin(synthetic_scene, (21, 22, 23))),
            np.flatnonzero(official_fold == 4),
        ),
    }


def domain_matrix(
    synthetic_feature: np.ndarray,
    official_feature: np.ndarray,
    synthetic_rows: np.ndarray,
    official_rows: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature = np.concatenate(
        [
            synthetic_feature[np.ix_(synthetic_rows, columns)],
            official_feature[np.ix_(official_rows, columns)],
        ]
    ).astype(np.float32)
    label = np.r_[
        np.zeros(len(synthetic_rows), dtype=np.int8),
        np.ones(len(official_rows), dtype=np.int8),
    ]
    return feature, label


def classifier_audit(
    synthetic: dict[str, np.ndarray],
    official: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], ExtraTreesClassifier, dict[str, np.ndarray]]:
    splits = domain_split(synthetic["scene"], official["fold"])
    selected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, (name, (synthetic_rows, official_rows)) in enumerate(splits.items()):
        selected[name] = balanced_domain_rows(
            synthetic_rows, official_rows, 379_000 + index
        )
    rows: dict[str, dict[str, float]] = {}
    all_model: ExtraTreesClassifier | None = None
    groups = feature_groups()
    for group, columns in groups.items():
        train_x, train_y = domain_matrix(
            synthetic["feature"], official["feature"], *selected["train"], columns
        )
        validation_x, validation_y = domain_matrix(
            synthetic["feature"], official["feature"], *selected["validation"], columns
        )
        test_x, test_y = domain_matrix(
            synthetic["feature"], official["feature"], *selected["test"], columns
        )
        linear = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=37901),
        )
        linear.fit(train_x, train_y)
        tree = ExtraTreesClassifier(
            n_estimators=DOMAIN_TREES,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=37902,
        )
        tree.fit(train_x, train_y)
        rows[group] = {
            "feature_count": int(len(columns)),
            "train_rows_per_domain": int(len(selected["train"][0])),
            "validation_linear_auc": float(
                roc_auc_score(validation_y, linear.predict_proba(validation_x)[:, 1])
            ),
            "validation_tree_auc": float(
                roc_auc_score(validation_y, tree.predict_proba(validation_x)[:, 1])
            ),
            "test_linear_auc": float(
                roc_auc_score(test_y, linear.predict_proba(test_x)[:, 1])
            ),
            "test_tree_auc": float(
                roc_auc_score(test_y, tree.predict_proba(test_x)[:, 1])
            ),
        }
        if group == "all":
            all_model = tree
    if all_model is None:
        raise RuntimeError("all-feature domain model was not trained")
    selected_array = {
        f"{split}_{domain}": rows_value[index]
        for split, rows_value in selected.items()
        for index, domain in enumerate(("synthetic", "official"))
    }
    return rows, all_model, selected_array


def marginal_shift(
    synthetic_feature: np.ndarray,
    official_feature: np.ndarray,
    synthetic_rows: np.ndarray,
    official_rows: np.ndarray,
) -> list[dict[str, float | int | str]]:
    synthetic_train = synthetic_feature[synthetic_rows].astype(np.float64)
    official_train = official_feature[official_rows].astype(np.float64)
    scale = np.std(synthetic_train, axis=0)
    scale = np.maximum(scale, 1e-4)
    dimensions = lgf.feature_dimensions()
    boundaries = (
        ("point", 0, dimensions["point"]),
        ("path", dimensions["point"], dimensions["point"] + dimensions["path"]),
        (
            "support",
            dimensions["point"] + dimensions["path"],
            dimensions["point"] + dimensions["path"] + dimensions["support"],
        ),
        (
            "terminal_ray",
            dimensions["point"] + dimensions["path"] + dimensions["support"],
            dimensions["total"],
        ),
    )
    group_of = np.empty(dimensions["total"], dtype=object)
    local_index = np.zeros(dimensions["total"], dtype=np.int64)
    for group, start, end in boundaries:
        group_of[start:end] = group
        local_index[start:end] = np.arange(end - start)
    rows = []
    for column in range(dimensions["total"]):
        shift = wasserstein_distance(
            synthetic_train[:, column], official_train[:, column]
        ) / scale[column]
        rows.append(
            {
                "column": column,
                "group": str(group_of[column]),
                "group_index": int(local_index[column]),
                "standardized_wasserstein": float(shift),
                "synthetic_mean": float(np.mean(synthetic_train[:, column])),
                "official_mean": float(np.mean(official_train[:, column])),
                "synthetic_std": float(np.std(synthetic_train[:, column])),
                "official_std": float(np.std(official_train[:, column])),
            }
        )
    return sorted(rows, key=lambda row: row["standardized_wasserstein"], reverse=True)


def density_ratio_audit(
    model: ExtraTreesClassifier,
    synthetic: dict[str, np.ndarray],
    official: dict[str, np.ndarray],
    selected: dict[str, np.ndarray],
) -> dict[str, object]:
    columns = feature_groups()["all"]
    synthetic_rows = selected["test_synthetic"]
    official_rows = selected["test_official"]
    synthetic_probability = model.predict_proba(
        synthetic["feature"][np.ix_(synthetic_rows, columns)]
    )[:, 1]
    official_probability = model.predict_proba(
        official["feature"][np.ix_(official_rows, columns)]
    )[:, 1]
    ratio = np.clip(
        synthetic_probability / np.maximum(1.0 - synthetic_probability, 1e-4),
        0.01,
        20.0,
    )
    effective = float(np.sum(ratio) ** 2 / np.sum(ratio**2))
    return {
        "synthetic_test_rows": int(len(ratio)),
        "effective_sample_size": effective,
        "effective_sample_fraction": effective / len(ratio),
        "weight_percentiles": {
            str(percentile): float(np.percentile(ratio, percentile))
            for percentile in (0, 10, 25, 50, 75, 90, 99, 100)
        },
        "domain_probability": {
            "synthetic_test_mean": float(np.mean(synthetic_probability)),
            "official_test_mean": float(np.mean(official_probability)),
            "synthetic_official_like_fraction_p_gt_0_5": float(
                np.mean(synthetic_probability > 0.5)
            ),
            "official_correct_fraction_p_gt_0_5": float(
                np.mean(official_probability > 0.5)
            ),
        },
    }


def main() -> None:
    for path in (
        TARGET_CACHE,
        SYNTHETIC_CACHE,
        v297.COMPONENT_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
        v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    synthetic_cache = np.load(SYNTHETIC_CACHE)
    synthetic = {key: synthetic_cache[key] for key in synthetic_cache.files}
    official = build_official_features()
    classifier_rows, domain_model, selected = classifier_audit(synthetic, official)
    shifts = marginal_shift(
        synthetic["feature"],
        official["feature"],
        selected["train_synthetic"],
        selected["train_official"],
    )
    density = density_ratio_audit(domain_model, synthetic, official, selected)
    report = {
        "method": VERSION,
        "status": "analysis_only",
        "scope": "simulation-to-official feature-domain audit; no prediction/full-H/ZIP",
        "primary_hypothesis": (
            "Direct lifecycle transfer failed because one or more point/path/support/ray "
            "feature groups have a measurable synthetic-to-official covariate shift."
        ),
        "dataset": {
            "synthetic_rows": int(len(synthetic["feature"])),
            "synthetic_scenes": int(len(np.unique(synthetic["scene"]))),
            "official_rows": int(len(official["feature"])),
            "official_folds": int(len(np.unique(official["fold"]))),
        },
        "domain_classifier": classifier_rows,
        "density_ratio": density,
        "top_marginal_shifts": shifts[:30],
        "group_mean_standardized_wasserstein": {
            group: float(
                np.mean(
                    [
                        row["standardized_wasserstein"]
                        for row in shifts
                        if row["group"] == group
                    ]
                )
            )
            for group in ("point", "path", "support", "terminal_ray")
        },
        "outputs": {
            "official_feature_cache": PATHS.relative(OFFICIAL_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["domain_classifier"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["density_ratio"], ensure_ascii=False, indent=2), flush=True)
    print(
        json.dumps(
            report["group_mean_standardized_wasserstein"],
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
