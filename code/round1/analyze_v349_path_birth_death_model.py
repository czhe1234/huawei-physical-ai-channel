#!/usr/bin/env python3
"""Predict candidate path birth/death before rendering official global PDP."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v328_patch_existence_support_field as v328
import analyze_v348_patchwise_path_energy_field as v348
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 349
VERSION = "v349_path_birth_death_model"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v349_path_birth_death_oof.npz"

ALPHA_GRID = np.linspace(0.0, 1.25, 11, dtype=np.float32)
PROBABILITY_THRESHOLDS = (0.0, 0.15, 0.30, 0.50, 0.70)
MAX_NEGATIVE_RATIO = 8
MIN_OFFICIAL_DELTA = 0.01
RANDOM_SEED = 349


def support_matrix(
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    label: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    xy: np.ndarray,
) -> np.ndarray:
    point, candidate_id, score, _truth = v328.predict(
        candidate, valid, label, fit_ids, query_ids, xy
    )
    output = np.zeros((len(candidate["score"]), candidate["score"].shape[1]), dtype=np.float32)
    output[point, candidate_id.astype(np.int64)] = score
    return output[query_ids]


def crossfit_support(
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    label: np.ndarray,
    fit_ids: np.ndarray,
    folds: list[np.ndarray],
    xy: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(fit_ids), candidate["score"].shape[1]), dtype=np.float32)
    lookup = {int(point): row for row, point in enumerate(fit_ids)}
    covered: list[int] = []
    for fold_ids in folds:
        query = np.intersect1d(fit_ids, fold_ids)
        if len(query) == 0:
            continue
        bank = np.setdiff1d(fit_ids, query)
        value = support_matrix(candidate, valid, label, bank, query, xy)
        rows = np.asarray([lookup[int(point)] for point in query], dtype=np.int64)
        output[rows] = value
        covered.extend(query.tolist())
    if len(np.unique(covered)) != len(fit_ids):
        raise AssertionError("cross-fit support did not cover all fit points")
    return output


def sample_anchor_energy(
    anchor: np.ndarray, row: np.ndarray, delay: np.ndarray
) -> np.ndarray:
    lower = np.floor(delay).astype(np.int64) % v297.N_DELAY
    upper = (lower + 1) % v297.N_DELAY
    fraction = delay - np.floor(delay)
    return (
        (1.0 - fraction) * anchor[row, lower]
        + fraction * anchor[row, upper]
    ).astype(np.float32)


def pair_features(
    point_ids: np.ndarray,
    candidate: dict[str, np.ndarray],
    support: np.ndarray,
    anchor: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = candidate["score"][point_ids] > 0.0
    row, candidate_id = np.nonzero(valid)
    point = point_ids[row]
    score = candidate["score"][point, candidate_id]
    length = candidate["path_length"][point, candidate_id]
    excess = candidate["excess_length"][point, candidate_id]
    delay = intercept + slope * length
    reflection = candidate["reflection_point"][point, candidate_id]
    relative = reflection - positions[point]
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    bs_distance = np.linalg.norm(positions[point] - bs[None, :], axis=1)
    anchor_energy = sample_anchor_energy(anchor, row, delay)
    angle = 2.0 * np.pi * delay / v297.N_DELAY
    features = np.column_stack(
        [
            positions[point, 0] / 300.0,
            positions[point, 1] / 300.0,
            positions[point, 2] / 30.0,
            bs_distance / 300.0,
            np.log1p(np.maximum(blocked[point], 0.0)) / 6.0,
            candidate_id.astype(np.float32) / candidate["score"].shape[1],
            score,
            length / 500.0,
            np.log1p(np.maximum(excess, 0.0)) / 6.0,
            relative[:, 0] / 300.0,
            relative[:, 1] / 300.0,
            relative[:, 2] / 30.0,
            np.sin(angle),
            np.cos(angle),
            support[row, candidate_id],
            np.sqrt(np.maximum(anchor_energy, 0.0)),
            np.log1p(anchor_energy * 1e4) / 8.0,
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32), row, candidate_id


def fit_models(features: np.ndarray, mass: np.ndarray, seed: int):
    positive = mass > 1e-6
    positive_ids = np.flatnonzero(positive)
    negative_ids = np.flatnonzero(~positive)
    rng = np.random.default_rng(seed)
    negative_take = min(len(negative_ids), MAX_NEGATIVE_RATIO * max(len(positive_ids), 1))
    selected_negative = rng.choice(negative_ids, size=negative_take, replace=False)
    selected = np.concatenate([positive_ids, selected_negative])
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.3,
        random_state=seed,
    )
    classifier.fit(features[selected], positive[selected].astype(np.int8))
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=140,
        max_leaf_nodes=24,
        min_samples_leaf=20,
        l2_regularization=0.5,
        random_state=seed + 1,
    )
    regressor.fit(features[positive_ids], np.log(np.maximum(mass[positive_ids], 1e-6)))
    return classifier, regressor, {
        "pair_count": int(len(mass)),
        "positive_count": int(len(positive_ids)),
        "negative_train_count": int(negative_take),
    }


def predict_mass(classifier, regressor, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = classifier.predict_proba(features)[:, 1].astype(np.float32)
    positive_mass = np.exp(regressor.predict(features)).astype(np.float32)
    mass = np.clip(probability * positive_mass, 0.0, 0.50).astype(np.float32)
    return probability, mass


def train_and_predict(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    folds: list[np.ndarray],
    candidate: dict[str, np.ndarray],
    candidate_valid: np.ndarray,
    reflection_label: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    fit_anchor: np.ndarray,
    query_anchor: np.ndarray,
    slope: float,
    intercept: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    fit_support = crossfit_support(
        candidate, candidate_valid, reflection_label, fit_ids, folds, positions[:, :2]
    )
    query_support = support_matrix(
        candidate,
        candidate_valid,
        reflection_label,
        fit_ids,
        query_ids,
        positions[:, :2],
    )
    fit_features, fit_row, fit_candidate = pair_features(
        fit_ids, candidate, fit_support, fit_anchor, positions, blocked, slope, intercept
    )
    query_features, query_row, query_candidate = pair_features(
        query_ids,
        candidate,
        query_support,
        query_anchor,
        positions,
        blocked,
        slope,
        intercept,
    )
    fit_mass = reflection_label[fit_ids[fit_row], fit_candidate]
    classifier, regressor, training = fit_models(fit_features, fit_mass, seed)
    probability, mass = predict_mass(classifier, regressor, query_features)
    probability_matrix = np.zeros((len(query_ids), candidate["score"].shape[1]), dtype=np.float32)
    mass_matrix = np.zeros_like(probability_matrix)
    probability_matrix[query_row, query_candidate] = probability
    mass_matrix[query_row, query_candidate] = mass
    return probability_matrix, mass_matrix, training


def render(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    positions: np.ndarray,
    candidate: dict[str, np.ndarray],
    direct_mass: np.ndarray,
    probability: np.ndarray,
    mass: np.ndarray,
    threshold: float,
    anchor: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, dict[str, object]]:
    tree = cKDTree(positions[fit_ids, :2])
    output = np.empty_like(anchor)
    selected_count = np.zeros(len(query_ids), dtype=np.float32)
    selected_mass = np.zeros(len(query_ids), dtype=np.float32)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    valid = candidate["score"] > 0.0
    for row, point_id in enumerate(query_ids):
        distance, local = tree.query(positions[point_id, :2], k=min(16, len(fit_ids)))
        distance = np.atleast_1d(distance)
        local = np.atleast_1d(local)
        direct = v348.weighted_mean(
            distance, direct_mass[fit_ids[local]].astype(np.float64), 2.0
        )
        active = np.flatnonzero(valid[point_id] & (probability[row] >= threshold))
        delays = [intercept + slope * float(np.linalg.norm(positions[point_id] - bs))]
        masses = [max(direct, 0.0)]
        for candidate_id in active:
            value = float(mass[row, candidate_id])
            if value <= 1e-7:
                continue
            delays.append(
                intercept + slope * float(candidate["path_length"][point_id, candidate_id])
            )
            masses.append(value)
        masses_array = np.asarray(masses, dtype=np.float32)
        total = float(np.sum(masses_array))
        if total > 0.95:
            masses_array *= np.float32(0.95 / total)
            total = 0.95
        atoms = v297.dirichlet_atoms(np.asarray(delays, dtype=np.float32)).T
        path = np.sum(masses_array[:, None] * atoms, axis=0)
        output[row] = v297.normalize_rows(
            ((1.0 - total) * anchor[row] + path)[None, :]
        )[0]
        selected_count[row] = max(len(masses) - 1, 0)
        selected_mass[row] = total
    return output, {
        "selected_reflections": summarize(selected_count),
        "path_mass": summarize(selected_mass),
        "active_fraction": float(np.mean(selected_count > 0.0)),
    }


def select_inner(anchor, truth, variants):
    rows = []
    for threshold, field in variants.items():
        for alpha in ALPHA_GRID:
            prediction = v348.blended(anchor, field, float(alpha))
            rows.append(
                {
                    "threshold": float(threshold),
                    "alpha": float(alpha),
                    "pdp": float(np.mean(v297.row_cos(prediction, truth))),
                }
            )
    return max(rows, key=lambda item: item["pdp"]), rows


def main() -> None:
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    target = np.load(v297.TARGET_CACHE)
    truth_d = v297.normalize_rows(target["d"].astype(np.float32))
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        target["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "train_blocked"
    ].astype(np.float32)
    folds = v315.disjoint_folds(valid_ids, positions[:, :2])
    data = np.load(v306.TRAIN_CANDIDATES)
    candidate = {key: data[key] for key in data.files}
    candidate_valid, reflection_label = v328.build_labels(candidate)
    labels, quality, _label_report = v306.build_labels()
    direct_mass = labels[:, 0] * quality
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    slope = float(calibration["slope_bins_per_meter"])
    intercept = float(calibration["intercept_bins"])

    rows = []
    ids_parts = []
    prediction_parts = []
    anchor_parts = []
    truth_parts = []
    for fold, outer_ids in enumerate(folds):
        inner_ids = folds[(fold + 1) % len(folds)]
        inner_fit = np.setdiff1d(valid_ids, np.concatenate([outer_ids, inner_ids]))
        inner_fit_anchor = v306.knn_anchor(inner_fit, inner_fit, positions, truth_d)
        inner_anchor = v306.knn_anchor(inner_ids, inner_fit, positions, truth_d)
        inner_probability, inner_mass, inner_training = train_and_predict(
            inner_fit,
            inner_ids,
            folds,
            candidate,
            candidate_valid,
            reflection_label,
            positions,
            blocked,
            inner_fit_anchor,
            inner_anchor,
            slope,
            intercept,
            RANDOM_SEED + fold,
        )
        inner_variants = {}
        for threshold in PROBABILITY_THRESHOLDS:
            inner_variants[threshold], _ = render(
                inner_ids,
                inner_fit,
                positions,
                candidate,
                direct_mass,
                inner_probability,
                inner_mass,
                threshold,
                inner_anchor,
                slope,
                intercept,
            )
        selected, inner_scan = select_inner(
            inner_anchor, truth_d[inner_ids], inner_variants
        )

        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        outer_fit_anchor = v306.knn_anchor(outer_fit, outer_fit, positions, truth_d)
        outer_anchor = v306.knn_anchor(outer_ids, outer_fit, positions, truth_d)
        probability, mass, outer_training = train_and_predict(
            outer_fit,
            outer_ids,
            folds,
            candidate,
            candidate_valid,
            reflection_label,
            positions,
            blocked,
            outer_fit_anchor,
            outer_anchor,
            slope,
            intercept,
            RANDOM_SEED + 100 + fold,
        )
        field, diagnostics = render(
            outer_ids,
            outer_fit,
            positions,
            candidate,
            direct_mass,
            probability,
            mass,
            float(selected["threshold"]),
            outer_anchor,
            slope,
            intercept,
        )
        prediction = v348.blended(outer_anchor, field, float(selected["alpha"]))
        base_score = v297.row_cos(outer_anchor, truth_d[outer_ids])
        model_score = v297.row_cos(prediction, truth_d[outer_ids])
        delta = model_score - base_score
        regimes = regime_of(blocked[outer_ids]).astype(str)
        rows.append(
            {
                "fold": fold,
                "selected_threshold": selected["threshold"],
                "selected_alpha": selected["alpha"],
                "inner_best_pdp": selected["pdp"],
                "inner_training": inner_training,
                "outer_training": outer_training,
                "anchor": summarize(base_score),
                "prediction": summarize(model_score),
                "delta": summarize(delta),
                "positive_fraction": float(np.mean(delta > 0.0)),
                "by_regime": {
                    regime: summarize(delta[regimes == regime])
                    for regime in ("clear", "light", "medium", "heavy")
                    if np.any(regimes == regime)
                },
                "diagnostics": diagnostics,
            }
        )
        ids_parts.append(outer_ids)
        prediction_parts.append(prediction)
        anchor_parts.append(outer_anchor)
        truth_parts.append(truth_d[outer_ids])
        print(
            f"v349 fold={fold} threshold={selected['threshold']:.2f} "
            f"alpha={selected['alpha']:.3f} delta={float(np.mean(delta)):+.6f}",
            flush=True,
        )

    fold_delta = np.asarray([row["delta"]["mean"] for row in rows], dtype=np.float64)
    mean_delta = float(np.mean(fold_delta))
    wins = int(np.sum(fold_delta > 0.0))
    passed = mean_delta >= MIN_OFFICIAL_DELTA and wins >= 4
    np.savez_compressed(
        OOF_OUT,
        ids=np.concatenate(ids_parts),
        anchor_d=np.concatenate(anchor_parts),
        prediction_d=np.concatenate(prediction_parts),
        truth_d=np.concatenate(truth_parts),
        fold=np.concatenate(
            [np.full(len(ids), fold, dtype=np.int16) for fold, ids in enumerate(ids_parts)]
        ),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "round": 2,
        "primary_hypothesis": (
            "A two-stage candidate birth/death classifier and positive-energy regressor, "
            "conditioned on fit-only patch support and anchor energy, can avoid v348's "
            "unconditional local averaging."
        ),
        "compliance": {
            "uses": [
                "official train H path matches and PDP",
                "official coordinates and Round1_Map candidates",
            ],
            "does_not_use": ["v207/v212/synthetic test truth", "online labels"],
        },
        "parameters": {
            "probability_thresholds": list(PROBABILITY_THRESHOLDS),
            "alpha_grid": ALPHA_GRID.tolist(),
            "nested_selection": "next disjoint fold",
        },
        "fold_rows": rows,
        "selection": {
            "minimum_delta": MIN_OFFICIAL_DELTA,
            "fold_delta": fold_delta.tolist(),
            "mean_delta": mean_delta,
            "wins": wins,
            "passed": bool(passed),
        },
        "outputs": {"oof": PATHS.relative(OOF_OUT), "report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["selection"], indent=2), flush=True)


if __name__ == "__main__":
    main()
