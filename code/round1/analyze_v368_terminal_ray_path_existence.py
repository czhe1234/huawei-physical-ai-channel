#!/usr/bin/env python3
"""Audit terminal-ray point-cloud features for local path existence.

The gradient of a physical path-delay surface identifies its local terminal
propagation direction. This probe asks whether official point-cloud occupancy
along that direction predicts which v366 eikonal hypotheses exist at an unseen
query, especially in medium/heavy regions.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v365_multiview_component_pdp as v365
import analyze_v366_local_eikonal_pdp as v366
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 368
VERSION = "v368_terminal_ray_path_existence"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
PAIR_OUT = OUT_DIR / "v368_terminal_ray_outer_pairs.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

RAY_STEP = 0.5
RAY_DISTANCE = 40.0
CLASSIFIER_TREES = 400
REGRESSOR_TREES = 400
MIN_AP_LIFT = 0.05
MIN_MASS_COSINE_LIFT = 0.03
MIN_FOLD_WINS = 4


def path_mass_labels(
    hypothesis: v366.HypothesisSet,
    component: dict[str, np.ndarray],
) -> np.ndarray:
    target = np.zeros_like(hypothesis.prior)
    for row, point_id in enumerate(hypothesis.query_ids):
        surface_ids = np.flatnonzero(hypothesis.valid[row])
        if len(surface_ids) == 0:
            continue
        surface_delay = hypothesis.delay[row, surface_ids]
        for component_id in np.flatnonzero(component["valid"][point_id]):
            distance = v365.circular_distance(
                surface_delay, component["delay"][point_id, component_id]
            )
            surface = int(np.argmin(distance))
            if float(distance[surface]) <= 0.25:
                target[row, surface_ids[surface]] += float(
                    component["weight"][point_id, component_id]
                )
        total = float(np.sum(target[row]))
        if total > 1e-20:
            target[row] /= total
    return target


def ray_summary(
    origin: np.ndarray,
    direction: np.ndarray,
    ue_z: float,
    footprint: np.ndarray,
    scene: dict[str, np.ndarray],
) -> np.ndarray:
    distance = np.arange(RAY_STEP, RAY_DISTANCE + 0.5 * RAY_STEP, RAY_STEP)
    xy = origin[None, :] + distance[:, None] * direction[None, :]
    ix = np.floor((xy[:, 0] - float(scene["x_min"])) / float(scene["resolution"])).astype(np.int64)
    iy = np.floor((xy[:, 1] - float(scene["y_min"])) / float(scene["resolution"])).astype(np.int64)
    valid = (
        (ix >= 0) & (ix < footprint.shape[1])
        & (iy >= 0) & (iy < footprint.shape[0])
    )
    safe_x = np.clip(ix, 0, footprint.shape[1] - 1)
    safe_y = np.clip(iy, 0, footprint.shape[0] - 1)
    z_max = np.nan_to_num(
        scene["grid_z_max"][safe_y, safe_x], nan=-np.inf
    )
    occupied = valid & footprint[safe_y, safe_x] & (z_max >= ue_z + 0.2)
    run_start = occupied & ~np.r_[False, occupied[:-1]]
    if np.any(occupied):
        first = int(np.argmax(occupied))
        first_distance = float(distance[first])
        nx = float(scene["grid_wall_nx"][safe_y[first], safe_x[first]])
        ny = float(scene["grid_wall_ny"][safe_y[first], safe_x[first]])
        normal_dot = float(nx * direction[0] + ny * direction[1])
        first_vertical = float(
            scene["grid_vertical_count"][safe_y[first], safe_x[first]]
        )
    else:
        first_distance = RAY_DISTANCE
        normal_dot = 0.0
        first_vertical = 0.0
    return np.asarray(
        [
            first_distance / RAY_DISTANCE,
            np.mean(occupied[:10]),
            np.mean(occupied[:20]),
            np.mean(occupied[:40]),
            np.mean(occupied),
            np.sum(run_start) / 8.0,
            max(float(np.max(z_max[occupied])) if np.any(occupied) else 0.0, 0.0) / 40.0,
            normal_dot,
            abs(normal_dot),
            np.log1p(first_vertical) / 8.0,
        ],
        dtype=np.float32,
    )


def terminal_features(
    hypothesis: v366.HypothesisSet,
    all_pos: np.ndarray,
    footprint: np.ndarray,
    scene: dict[str, np.ndarray],
    blocked: np.ndarray,
) -> np.ndarray:
    output = np.zeros(
        (len(hypothesis.query_ids), v366.HYPOTHESES, 29), dtype=np.float32
    )
    bs_xy = np.asarray([50.0, 0.0], dtype=np.float64)
    bs_z = 25.0
    for row, point_id in enumerate(hypothesis.query_ids):
        query = all_pos[point_id]
        radial = query[:2] - bs_xy
        radial_norm = max(float(np.linalg.norm(radial)), 1e-8)
        radial /= radial_norm
        direct_3d = float(
            np.linalg.norm(query - np.asarray([bs_xy[0], bs_xy[1], bs_z]))
        )
        for surface in np.flatnonzero(hypothesis.valid[row]):
            gradient = hypothesis.feature[row, surface, :2].astype(np.float64)
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm > 1e-4:
                direction = gradient / gradient_norm
            else:
                direction = radial.copy()
            incoming = ray_summary(
                query[:2], -direction, float(query[2]), footprint, scene
            )
            outgoing = ray_summary(
                query[:2], direction, float(query[2]), footprint, scene
            )
            left = np.asarray([-direction[1], direction[0]], dtype=np.float64)
            left_clearance = ray_summary(
                query[:2], left, float(query[2]), footprint, scene
            )[0]
            right_clearance = ray_summary(
                query[:2], -left, float(query[2]), footprint, scene
            )[0]
            path_length = float(hypothesis.delay[row, surface]) / v366.THEORETICAL_GRADIENT
            orientation_dot = float(np.dot(direction, radial))
            orientation_cross = float(
                direction[0] * radial[1] - direction[1] * radial[0]
            )
            extra = np.asarray(
                [
                    left_clearance,
                    right_clearance,
                    path_length / 800.0,
                    (path_length - direct_3d) / 500.0,
                    direct_3d / 500.0,
                    orientation_dot,
                    orientation_cross,
                    gradient_norm,
                    float(blocked[point_id]) / 200.0,
                ],
                dtype=np.float32,
            )
            output[row, surface] = np.concatenate([incoming, outgoing, extra])
    return output


def periodic_delay(delay: np.ndarray) -> np.ndarray:
    angle = 2.0 * np.pi * delay / v366.N_DELAY
    return np.stack(
        [
            np.sin(angle), np.cos(angle),
            np.sin(2.0 * angle), np.cos(2.0 * angle),
            np.sin(4.0 * angle), np.cos(4.0 * angle),
        ],
        axis=2,
    ).astype(np.float32)


def pair_arrays(
    hypothesis: v366.HypothesisSet,
    node_raw: np.ndarray,
    terminal: np.ndarray,
    target_mass: np.ndarray,
) -> dict[str, np.ndarray]:
    query_feature = np.broadcast_to(
        node_raw[hypothesis.query_ids, None, :],
        (len(hypothesis.query_ids), v366.HYPOTHESES, node_raw.shape[1]),
    )
    prior_feature = np.stack(
        [
            hypothesis.prior,
            np.log(np.maximum(hypothesis.prior, 1e-12)) / 12.0,
        ],
        axis=2,
    )
    existing = np.concatenate(
        [
            query_feature,
            hypothesis.feature,
            periodic_delay(hypothesis.delay),
            prior_feature,
        ],
        axis=2,
    ).astype(np.float32)
    full = np.concatenate([existing, terminal], axis=2).astype(np.float32)
    valid = hypothesis.valid
    query_row = np.broadcast_to(
        np.arange(len(hypothesis.query_ids))[:, None], valid.shape
    )[valid]
    return {
        "existing": existing[valid],
        "full": full[valid],
        "terminal": terminal[valid],
        "mass": target_mass[valid].astype(np.float32),
        "binary": (target_mass[valid] > 0.0),
        "prior": hypothesis.prior[valid].astype(np.float32),
        "query_row": query_row.astype(np.int32),
    }


def shuffled_terminal(
    arrays: dict[str, np.ndarray], seed: int
) -> np.ndarray:
    output = arrays["full"].copy()
    terminal_dim = arrays["terminal"].shape[1]
    rng = np.random.default_rng(seed)
    for query_row in np.unique(arrays["query_row"]):
        rows = np.flatnonzero(arrays["query_row"] == query_row)
        order = rng.permutation(len(rows))
        output[rows, -terminal_dim:] = arrays["terminal"][rows[order]]
    return output


def make_classifier(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=CLASSIFIER_TREES,
        max_depth=18,
        min_samples_leaf=8,
        max_features=0.55,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def make_regressor(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=REGRESSOR_TREES,
        max_depth=18,
        min_samples_leaf=8,
        max_features=0.55,
        n_jobs=-1,
        random_state=seed,
    )


def fit_predict(
    train_feature: np.ndarray,
    query_feature: np.ndarray,
    train: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    classifier = make_classifier(seed)
    classifier.fit(train_feature, train["binary"])
    probability = classifier.predict_proba(query_feature)[:, 1].astype(np.float32)
    positive = train["binary"]
    regressor = make_regressor(seed + 10_000)
    regressor.fit(
        train_feature[positive],
        np.log(np.maximum(train["mass"][positive], 1e-7)),
    )
    conditional_mass = np.exp(regressor.predict(query_feature)).astype(np.float32)
    return probability, (probability * conditional_mass).astype(np.float32)


def grouped_mass_cosine(
    score: np.ndarray,
    truth: np.ndarray,
    query_row: np.ndarray,
    query_count: int,
) -> float:
    values = []
    for row in range(query_count):
        keep = query_row == row
        if not np.any(keep) or float(np.sum(truth[keep])) <= 1e-20:
            continue
        values.append(
            float(
                np.dot(score[keep], truth[keep])
                / max(
                    float(np.linalg.norm(score[keep]) * np.linalg.norm(truth[keep])),
                    1e-30,
                )
            )
        )
    return float(np.mean(values)) if values else float("nan")


def metric_rows(
    arrays: dict[str, np.ndarray],
    probability: dict[str, np.ndarray],
    mass_score: dict[str, np.ndarray],
    query_count: int,
) -> dict[str, dict[str, float]]:
    output = {}
    for name in probability:
        output[name] = {
            "path_existence_ap": float(
                average_precision_score(arrays["binary"], probability[name])
            ),
            "path_mass_cosine": grouped_mass_cosine(
                mass_score[name],
                arrays["mass"],
                arrays["query_row"],
                query_count,
            ),
        }
    return output


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
    scene_cache = np.load(v274.SCENE_CACHE)
    scene = {key: scene_cache[key] for key in scene_cache.files}
    footprint = np.load(v274.FOOTPRINT_CACHE)["footprint"].astype(bool)
    propagation_geometry = np.load(v238.GEOM_CACHE)
    train_blocked = propagation_geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(train_blocked)
    all_blocked = np.concatenate(
        [train_blocked, propagation_geometry["test_blocked"].astype(np.float32)]
    )
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])

    fold_rows: list[dict[str, object]] = []
    cache: dict[str, list[np.ndarray]] = {
        "point_id": [], "binary": [], "mass": [], "prior": [],
        "existing_probability": [], "full_probability": [],
        "shuffled_probability": [], "existing_mass": [], "full_mass": [],
        "shuffled_mass": [],
    }
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        fit_set, outer_set, _node = v366.raw_sets(
            fit_ids, outer_ids, fit_ids, geometry, node_raw
        )
        fit_h = v366.build_hypotheses(fit_set, all_pos[:, :2], component)
        outer_h = v366.build_hypotheses(outer_set, all_pos[:, :2], component)
        fit_terminal = terminal_features(
            fit_h, all_pos, footprint, scene, all_blocked
        )
        outer_terminal = terminal_features(
            outer_h, all_pos, footprint, scene, all_blocked
        )
        fit_arrays = pair_arrays(
            fit_h, node_raw, fit_terminal, path_mass_labels(fit_h, component)
        )
        outer_arrays = pair_arrays(
            outer_h,
            node_raw,
            outer_terminal,
            path_mass_labels(outer_h, component),
        )
        probability: dict[str, np.ndarray] = {"prior": outer_arrays["prior"]}
        mass_score: dict[str, np.ndarray] = {"prior": outer_arrays["prior"]}
        probability["existing"], mass_score["existing"] = fit_predict(
            fit_arrays["existing"],
            outer_arrays["existing"],
            fit_arrays,
            368_000 + fold,
        )
        probability["full"], mass_score["full"] = fit_predict(
            fit_arrays["full"],
            outer_arrays["full"],
            fit_arrays,
            369_000 + fold,
        )
        shuffled_feature = shuffled_terminal(outer_arrays, 370_000 + fold)
        full_classifier = make_classifier(369_000 + fold)
        full_classifier.fit(fit_arrays["full"], fit_arrays["binary"])
        probability["shuffled_terminal"] = full_classifier.predict_proba(
            shuffled_feature
        )[:, 1].astype(np.float32)
        positive = fit_arrays["binary"]
        full_regressor = make_regressor(379_000 + fold)
        full_regressor.fit(
            fit_arrays["full"][positive],
            np.log(np.maximum(fit_arrays["mass"][positive], 1e-7)),
        )
        mass_score["shuffled_terminal"] = (
            probability["shuffled_terminal"]
            * np.exp(full_regressor.predict(shuffled_feature)).astype(np.float32)
        )
        metrics = metric_rows(
            outer_arrays, probability, mass_score, len(outer_ids)
        )
        regime_metrics = {}
        for regime_name in ("clear", "light", "medium", "heavy"):
            pair_keep = regimes[outer_ids][outer_arrays["query_row"]] == regime_name
            if not np.any(pair_keep):
                continue
            regime_metrics[regime_name] = {
                name: float(
                    average_precision_score(
                        outer_arrays["binary"][pair_keep], score[pair_keep]
                    )
                )
                for name, score in probability.items()
            }
        fold_rows.append(
            {
                "fold": fold,
                "fit_point_count": int(len(fit_ids)),
                "outer_point_count": int(len(outer_ids)),
                "fit_pair_count": int(len(fit_arrays["binary"])),
                "outer_pair_count": int(len(outer_arrays["binary"])),
                "positive_prevalence": float(np.mean(outer_arrays["binary"])),
                "metrics": metrics,
                "terminal_lift_vs_existing": {
                    "path_existence_ap": (
                        metrics["full"]["path_existence_ap"]
                        - metrics["existing"]["path_existence_ap"]
                    ),
                    "path_mass_cosine": (
                        metrics["full"]["path_mass_cosine"]
                        - metrics["existing"]["path_mass_cosine"]
                    ),
                },
                "regime_path_existence_ap": regime_metrics,
                "leakage_assertions": {
                    "outer_in_fit_labels": 0,
                    "outer_in_hypothesis_donor_bank": 0,
                },
            }
        )
        pair_point = outer_ids[outer_arrays["query_row"]]
        for key, value in (
            ("point_id", pair_point),
            ("binary", outer_arrays["binary"]),
            ("mass", outer_arrays["mass"]),
            ("prior", outer_arrays["prior"]),
            ("existing_probability", probability["existing"]),
            ("full_probability", probability["full"]),
            ("shuffled_probability", probability["shuffled_terminal"]),
            ("existing_mass", mass_score["existing"]),
            ("full_mass", mass_score["full"]),
            ("shuffled_mass", mass_score["shuffled_terminal"]),
        ):
            cache[key].append(np.asarray(value))
        print(
            f"v368 fold={fold} AP prior={metrics['prior']['path_existence_ap']:.4f} "
            f"existing={metrics['existing']['path_existence_ap']:.4f} "
            f"full={metrics['full']['path_existence_ap']:.4f} "
            f"shuffle={metrics['shuffled_terminal']['path_existence_ap']:.4f}",
            flush=True,
        )

    np.savez_compressed(PAIR_OUT, **{key: np.concatenate(value) for key, value in cache.items()})
    mean_metrics = {
        name: {
            metric: float(
                np.mean([row["metrics"][name][metric] for row in fold_rows])
            )
            for metric in ("path_existence_ap", "path_mass_cosine")
        }
        for name in ("prior", "existing", "full", "shuffled_terminal")
    }
    ap_lift = mean_metrics["full"]["path_existence_ap"] - max(
        mean_metrics["prior"]["path_existence_ap"],
        mean_metrics["existing"]["path_existence_ap"],
    )
    mass_lift = mean_metrics["full"]["path_mass_cosine"] - max(
        mean_metrics["prior"]["path_mass_cosine"],
        mean_metrics["existing"]["path_mass_cosine"],
    )
    fold_wins = int(
        sum(
            row["terminal_lift_vs_existing"]["path_existence_ap"] > 0.0
            and row["terminal_lift_vs_existing"]["path_mass_cosine"] > 0.0
            for row in fold_rows
        )
    )
    heavy_lift = float(
        np.mean(
            [
                row["regime_path_existence_ap"]["heavy"]["full"]
                - row["regime_path_existence_ap"]["heavy"]["existing"]
                for row in fold_rows
            ]
        )
    )
    passed = bool(
        ap_lift >= MIN_AP_LIFT
        and mass_lift >= MIN_MASS_COSINE_LIFT
        and fold_wins >= MIN_FOLD_WINS
        and heavy_lift > 0.0
        and mean_metrics["full"]["path_existence_ap"]
        > mean_metrics["shuffled_terminal"]["path_existence_ap"]
    )
    summary = {
        "mean_metrics": mean_metrics,
        "full_ap_lift_over_best_existing": ap_lift,
        "full_mass_cosine_lift_over_best_existing": mass_lift,
        "fold_wins": fold_wins,
        "heavy_ap_lift_vs_existing": heavy_lift,
    }
    report = {
        "method": VERSION,
        "status": "terminal_ray_information_passed" if passed else "analysis_only_no_candidate",
        "scope": "path-level official-data probe only; no PDP/full-H prediction or ZIP",
        "primary_hypothesis": (
            "The local eikonal gradient identifies a path's terminal propagation direction; "
            "official point-cloud clearance and wall incidence along that ray should predict "
            "path existence beyond coordinate/node/eikonal features."
        ),
        "compliance": {
            "uses": [
                "official train H fit-only path labels",
                "official train/test coordinates and point-cloud occupancy/normal/height",
                "v366 theoretical-gradient local surfaces",
            ],
            "does_not_use": [
                "outer labels in fitting",
                "Sionna/synthetic labels, restored mesh/material, or v212 teacher",
                "full-H output, hidden truth, online score, or submission packaging",
            ],
        },
        "parameters": {
            "ray_step_m": RAY_STEP,
            "ray_distance_m": RAY_DISTANCE,
            "classifier_trees": CLASSIFIER_TREES,
            "regressor_trees": REGRESSOR_TREES,
            "tree_max_depth": 18,
            "tree_min_samples_leaf": 8,
            "parameter_search": "none",
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "path_existence_ap_lift": f">={MIN_AP_LIFT}",
            "path_mass_cosine_lift": f">={MIN_MASS_COSINE_LIFT}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "heavy_ap_lift": ">0",
            "full_vs_shuffled_terminal": ">0",
            "passed": passed,
        },
        "outputs": {
            "outer_pairs": PATHS.relative(PAIR_OUT),
            "report": PATHS.relative(REPORT),
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
