#!/usr/bin/env python3
"""Build v381 lifecycle examples with official-matched isolation folds."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v372_virtual_source_gap_reconnection as v372
import build_v382_dense_lifecycle_examples as v382
import lifecycle_geometry_features as lgf
import train_v375_synthetic_lifecycle_pretraining as v375
from project_paths import PATHS


VERSION_NUMBER = 383
VERSION = "v383_isolation_matched_lifecycle_examples"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLES = OUT_DIR / "v383_isolation_matched_lifecycle_examples.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
SCENE_ROOT = PATHS.cache_dir(381) / "scenes"
OFFICIAL_CACHE = PATHS.cache_dir(379) / "v379_official_lifecycle_features.npz"
V382_REPORT = PATHS.version_report_dir(382) / "v382_dense_lifecycle_examples.json"

SCENE_COUNT = 30
FOLD_COUNT = 5
STRATA = 20
FAMILIES_PER_SCENE = 300


def isolation_folds(position: np.ndarray, seed: int) -> list[np.ndarray]:
    ids = np.arange(len(position), dtype=np.int64)
    nearest = cKDTree(position[:, :2]).query(position[:, :2], k=2)[0][:, 1]
    edges = np.quantile(nearest, np.linspace(0.0, 1.0, STRATA + 1))
    stratum = np.clip(
        np.searchsorted(edges[1:-1], nearest, side="right"), 0, STRATA - 1
    )
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(FOLD_COUNT)]
    for value in range(STRATA):
        current = ids[stratum == value].copy()
        rng.shuffle(current)
        offset = value % FOLD_COUNT
        for index, point_id in enumerate(current):
            folds[(index + offset) % FOLD_COUNT].append(int(point_id))
    return [np.sort(np.asarray(fold, dtype=np.int64)) for fold in folds]


def build_examples() -> dict[str, np.ndarray]:
    if EXAMPLES.is_file():
        cached = np.load(EXAMPLES)
        return {key: cached[key] for key in cached.files}
    rng = np.random.default_rng(38301)
    rows: dict[str, list[np.ndarray]] = {
        "feature": [],
        "binary": [],
        "mass": [],
        "scene": [],
        "family": [],
        "group": [],
    }
    support_distance_rows = []
    for scene_id in range(SCENE_COUNT):
        data = np.load(
            SCENE_ROOT / f"scene_{scene_id:03d}" / "lifecycle.npz"
        )
        position = data["positions"].astype(np.float64)
        bs = data["bs"].astype(np.float64)
        valid = data["valid"].astype(bool)
        delay = data["delay_bin"].astype(np.float64)
        mass = data["normalized_power"].astype(np.float64)
        frequency = float(data["frequency"])
        raster = lgf.raster_from_boxes(data["boxes"].astype(np.float64))
        point_feature = lgf.point_features(raster, position, bs)
        folds = isolation_folds(position, 383_100 + scene_id)
        family_score = np.sum(mass, axis=0) * np.sqrt(
            np.maximum(np.sum(valid, axis=0), 1)
        )
        family_ids = np.argsort(family_score)[::-1][:FAMILIES_PER_SCENE]
        scene_feature = []
        scene_binary = []
        scene_mass = []
        scene_family = []
        scene_group = []
        for family_id in family_ids:
            if int(np.sum(valid[:, family_id])) < 8:
                continue
            for held_fold, query in enumerate(folds):
                support = np.flatnonzero(valid[:, family_id])
                support = np.setdiff1d(support, query, assume_unique=True)
                if len(support) < 4 or len(query) == 0:
                    continue
                support_mass = mass[support, family_id]
                model, fit_rmse = v375.fit_family_surface(
                    position[support], delay[support, family_id], support_mass
                )
                selected = v375.select_examples(
                    rng=rng,
                    query_ids=query,
                    valid=valid[:, family_id],
                    position=position,
                    support_xy=position[support, :2],
                )
                if len(selected) == 0:
                    continue
                predicted_delay, predicted_gradient = v372.affine_prediction(
                    model, position[selected, :2]
                )
                support_distance_rows.extend(
                    cKDTree(position[support, :2])
                    .query(position[selected, :2], k=1)[0]
                    .tolist()
                )
                group_id = scene_id * 1_000_000 + int(family_id) * 5 + held_fold
                for row, point_id in enumerate(selected):
                    scene_feature.append(
                        lgf.family_feature(
                            scene=raster,
                            point=position[point_id],
                            bs=bs,
                            point_feature=point_feature[point_id],
                            predicted_delay=float(predicted_delay[row]),
                            predicted_gradient=predicted_gradient[row],
                            support_xy=position[support, :2],
                            support_mass=support_mass,
                            fit_delay_rmse=fit_rmse,
                            frequency=frequency,
                        )
                    )
                    scene_binary.append(bool(valid[point_id, family_id]))
                    scene_mass.append(float(mass[point_id, family_id]))
                    scene_family.append(int(family_id))
                    scene_group.append(group_id)
        if not scene_feature:
            raise RuntimeError(f"scene {scene_id} generated no lifecycle examples")
        rows["feature"].append(np.stack(scene_feature).astype(np.float32))
        rows["binary"].append(np.asarray(scene_binary, dtype=np.int8))
        rows["mass"].append(np.asarray(scene_mass, dtype=np.float32))
        rows["scene"].append(np.full(len(scene_binary), scene_id, dtype=np.int8))
        rows["family"].append(np.asarray(scene_family, dtype=np.int32))
        rows["group"].append(np.asarray(scene_group, dtype=np.int64))
        print(
            f"v383 scene={scene_id} examples={len(scene_binary)} "
            f"positive={int(np.sum(scene_binary))}",
            flush=True,
        )
    output = {key: np.concatenate(value) for key, value in rows.items()}
    output["support_distance_audit"] = np.asarray(
        support_distance_rows, dtype=np.float32
    )
    np.savez_compressed(EXAMPLES, **output)
    return output


def main() -> None:
    for path in (OFFICIAL_CACHE, V382_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    synthetic = build_examples()
    official_cache = np.load(OFFICIAL_CACHE)
    official = {key: official_cache[key] for key in official_cache.files}
    audit = v382.domain_audit(synthetic, official)
    old = json.loads(V382_REPORT.read_text(encoding="utf-8"))
    comparison = {
        group: {
            "held_quadrant_test_tree_auc": float(
                old["domain_audit"]["classifiers"][group]["test_tree_auc"]
            ),
            "isolation_fold_test_tree_auc": float(
                audit["classifiers"][group]["test_tree_auc"]
            ),
            "auc_reduction": float(
                old["domain_audit"]["classifiers"][group]["test_tree_auc"]
                - audit["classifiers"][group]["test_tree_auc"]
            ),
        }
        for group in audit["classifiers"]
    }
    report = {
        "method": VERSION,
        "status": "isolation_matched_examples_built_domain_audited",
        "scope": "official-matched synthetic exposure interface and domain audit",
        "dataset": {
            "rows": int(len(synthetic["feature"])),
            "positive_rows": int(np.sum(synthetic["binary"])),
            "feature_dimension": int(synthetic["feature"].shape[1]),
            "train_scenes": list(v382.TRAIN_SCENES),
            "validation_scenes": list(v382.VALIDATION_SCENES),
            "test_scenes": list(v382.TEST_SCENES),
            "support_distance_mean_m": float(
                np.mean(synthetic["support_distance_audit"])
            ),
            "support_distance_percentiles_m": {
                str(percentile): float(
                    np.percentile(synthetic["support_distance_audit"], percentile)
                )
                for percentile in (0, 25, 50, 75, 90, 99, 100)
            },
        },
        "domain_audit": audit,
        "comparison_to_v382": comparison,
        "outputs": {
            "examples": PATHS.relative(EXAMPLES),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["dataset"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print(
        json.dumps(
            audit["group_mean_standardized_wasserstein"],
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
