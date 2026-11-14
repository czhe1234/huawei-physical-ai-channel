#!/usr/bin/env python3
"""Build v381 lifecycle examples and audit their official feature-domain gap."""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analyze_v372_virtual_source_gap_reconnection as v372
import analyze_v379_lifecycle_domain_gap as v379
import lifecycle_geometry_features as lgf
import train_v375_synthetic_lifecycle_pretraining as v375
from project_paths import PATHS


VERSION_NUMBER = 382
VERSION = "v382_dense_lifecycle_examples"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLES = OUT_DIR / "v382_dense_lifecycle_examples.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
SCENE_ROOT = PATHS.cache_dir(381) / "scenes"
OFFICIAL_CACHE = PATHS.cache_dir(379) / "v379_official_lifecycle_features.npz"
OLD_DOMAIN_REPORT = PATHS.version_report_dir(379) / "v379_lifecycle_domain_gap.json"

SCENE_COUNT = 30
FAMILIES_PER_SCENE = 300
TRAIN_SCENES = tuple(range(22))
VALIDATION_SCENES = (22, 23, 24, 25)
TEST_SCENES = (26, 27, 28, 29)
DOMAIN_TREES = 300
MAX_DOMAIN_ROWS_PER_SOURCE = 60000


def build_examples() -> dict[str, np.ndarray]:
    if EXAMPLES.is_file():
        cached = np.load(EXAMPLES)
        return {key: cached[key] for key in cached.files}
    rng = np.random.default_rng(38201)
    rows: dict[str, list[np.ndarray]] = {
        "feature": [],
        "binary": [],
        "mass": [],
        "scene": [],
        "family": [],
        "group": [],
    }
    for scene_id in range(SCENE_COUNT):
        path = SCENE_ROOT / f"scene_{scene_id:03d}" / "lifecycle.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.load(path)
        position = data["positions"].astype(np.float64)
        bs = data["bs"].astype(np.float64)
        valid = data["valid"].astype(bool)
        delay = data["delay_bin"].astype(np.float64)
        mass = data["normalized_power"].astype(np.float64)
        frequency = float(data["frequency"])
        raster = lgf.raster_from_boxes(data["boxes"].astype(np.float64))
        point_feature = lgf.point_features(raster, position, bs)
        fold = v375.spatial_fold(position)
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
            for held_fold in range(4):
                support = np.flatnonzero(valid[:, family_id] & (fold != held_fold))
                query = np.flatnonzero(fold == held_fold)
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
                group_id = scene_id * 1_000_000 + int(family_id) * 4 + held_fold
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
            f"v382 scene={scene_id} examples={len(scene_binary)} "
            f"positive={int(np.sum(scene_binary))}",
            flush=True,
        )
    output = {key: np.concatenate(value) for key, value in rows.items()}
    if output["feature"].shape[1] != lgf.feature_dimensions()["total"]:
        raise RuntimeError("v382 feature dimension mismatch")
    np.savez_compressed(EXAMPLES, **output)
    return output


def split_rows(
    synthetic_scene: np.ndarray,
    official_fold: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "train": (
            np.flatnonzero(np.isin(synthetic_scene, TRAIN_SCENES)),
            np.flatnonzero(np.isin(official_fold, (0, 1, 2))),
        ),
        "validation": (
            np.flatnonzero(np.isin(synthetic_scene, VALIDATION_SCENES)),
            np.flatnonzero(official_fold == 3),
        ),
        "test": (
            np.flatnonzero(np.isin(synthetic_scene, TEST_SCENES)),
            np.flatnonzero(official_fold == 4),
        ),
    }


def balanced_rows(
    synthetic_rows: np.ndarray,
    official_rows: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    take = min(
        len(synthetic_rows), len(official_rows), MAX_DOMAIN_ROWS_PER_SOURCE
    )
    return (
        rng.choice(synthetic_rows, size=take, replace=False),
        rng.choice(official_rows, size=take, replace=False),
    )


def domain_audit(
    synthetic: dict[str, np.ndarray],
    official: dict[str, np.ndarray],
) -> dict[str, object]:
    split = split_rows(synthetic["scene"], official["fold"])
    selected = {
        name: balanced_rows(*rows, 382_100 + index)
        for index, (name, rows) in enumerate(split.items())
    }
    groups = v379.feature_groups()
    result: dict[str, dict[str, float]] = {}
    for group, columns in groups.items():
        matrices = {}
        for split_name in ("train", "validation", "test"):
            matrices[split_name] = v379.domain_matrix(
                synthetic["feature"],
                official["feature"],
                *selected[split_name],
                columns,
            )
        linear = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=38211),
        )
        linear.fit(*matrices["train"])
        tree = ExtraTreesClassifier(
            n_estimators=DOMAIN_TREES,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=38212,
        )
        tree.fit(*matrices["train"])
        result[group] = {
            "validation_linear_auc": float(
                roc_auc_score(
                    matrices["validation"][1],
                    linear.predict_proba(matrices["validation"][0])[:, 1],
                )
            ),
            "validation_tree_auc": float(
                roc_auc_score(
                    matrices["validation"][1],
                    tree.predict_proba(matrices["validation"][0])[:, 1],
                )
            ),
            "test_linear_auc": float(
                roc_auc_score(
                    matrices["test"][1],
                    linear.predict_proba(matrices["test"][0])[:, 1],
                )
            ),
            "test_tree_auc": float(
                roc_auc_score(
                    matrices["test"][1],
                    tree.predict_proba(matrices["test"][0])[:, 1],
                )
            ),
        }
    shifts = v379.marginal_shift(
        synthetic["feature"],
        official["feature"],
        selected["train"][0],
        selected["train"][1],
    )
    group_shift = {
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
    }
    return {
        "classifiers": result,
        "group_mean_standardized_wasserstein": group_shift,
        "top_marginal_shifts": shifts[:30],
    }


def main() -> None:
    for path in (OFFICIAL_CACHE, OLD_DOMAIN_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    synthetic = build_examples()
    official_cache = np.load(OFFICIAL_CACHE)
    official = {key: official_cache[key] for key in official_cache.files}
    audit = domain_audit(synthetic, official)
    old = json.loads(OLD_DOMAIN_REPORT.read_text(encoding="utf-8"))
    comparison = {
        group: {
            "v374_test_tree_auc": float(
                old["domain_classifier"][group]["test_tree_auc"]
            ),
            "v381_test_tree_auc": float(
                audit["classifiers"][group]["test_tree_auc"]
            ),
            "auc_reduction": float(
                old["domain_classifier"][group]["test_tree_auc"]
                - audit["classifiers"][group]["test_tree_auc"]
            ),
        }
        for group in audit["classifiers"]
    }
    report = {
        "method": VERSION,
        "status": "dense_examples_built_domain_audited",
        "scope": "v381 scene-held 85D lifecycle examples and official domain audit",
        "dataset": {
            "rows": int(len(synthetic["feature"])),
            "positive_rows": int(np.sum(synthetic["binary"])),
            "feature_dimension": int(synthetic["feature"].shape[1]),
            "train_scenes": list(TRAIN_SCENES),
            "validation_scenes": list(VALIDATION_SCENES),
            "test_scenes": list(TEST_SCENES),
        },
        "domain_audit": audit,
        "comparison_to_v374": comparison,
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
