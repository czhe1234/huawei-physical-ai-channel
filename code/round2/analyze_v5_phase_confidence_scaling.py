#!/usr/bin/env python3
"""Learn pointwise carrier shrinkage from observable Round 2 phase confidence."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_v4_map_virtual_source_phase import METERS_PER_DELAY_BIN
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import (
    MH,
    MV,
    S_SC,
    TINY,
    ComponentCache,
    Config as TransportConfig,
    circular_delta,
    component_energy,
    match_components,
)
from run_rbf_spectral_field import OOF_META


VERSION = "Round2-E25-phase-confidence-scaling"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_phase_confidence_scaling"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_phase_confidence_scaling"
FEATURE_CACHE = ARTIFACT_DIR / "oof_phase_confidence_features.npz"
PREDICTION_CACHE = ARTIFACT_DIR / "oof_phase_confidence_scale_predictions.npz"
REPORT = ANALYSIS_DIR / "phase_confidence_scaling.json"

V4_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_v4_calibrated_exact_projection"
    / "calibrated_exact_projection.json"
)
V4_METRICS = (
    PATHS.artifacts
    / "round2"
    / "v4_calibrated_exact_projection"
    / "calibrated_exact_projection_metrics.npz"
)
GEOMETRY = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_geometry_visibility_features.npz"
)
DESCRIPTORS = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "oof_foundation_spectral_descriptors.npz"
)
OOF_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "v3_anchored_neural_rbf"
    / "oof_epoch6_corrected_weights.npz"
)
COMPONENT_CACHE = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
MAP_PHASE_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_v4_map_virtual_source_phase"
    / "map_virtual_source_phase.json"
)


@dataclass(frozen=True)
class Config:
    paths: int = 16
    path_donors: int = 24
    maximum_scale: float = 1.5
    extra_trees: int = 400
    extra_depth: int = 6
    extra_leaf: int = 12
    seed: int = 20_260_805


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def load_components() -> ComponentCache:
    cached = np.load(COMPONENT_CACHE)
    return ComponentCache(
        coordinates=cached["coordinates"].astype(np.float32),
        coefficients=cached["coefficients"].astype(np.complex64),
        residual_fraction=cached["residual_fraction"].astype(np.float32),
        point_power=cached["point_power"].astype(np.float64),
    )


def load_beta_table() -> dict[tuple[int, int], float]:
    payload = json.loads(MAP_PHASE_REPORT.read_text(encoding="utf-8"))
    return {
        (int(row["fold"]), int(row["site"])): float(row["beta_rad_per_m"])
        for row in payload["diagnostics"]["beta_calibration"]
    }


def selected_v4_arrays() -> dict[str, np.ndarray]:
    payload = json.loads(V4_REPORT.read_text(encoding="utf-8"))
    metrics = np.load(V4_METRICS)
    names = metrics["candidate_names"]
    fold = metrics["fold_id"].astype(np.int8)
    site = metrics["site"].astype(np.int8)
    count = len(fold)
    output = {
        "candidate_index": np.empty(count, dtype=np.int16),
        "fixed_scale": np.empty(count, dtype=np.float64),
        "predicted_energy": np.empty(count, dtype=np.float64),
        "truth_energy": np.empty(count, dtype=np.float64),
        "cross": np.empty(count, dtype=np.float64),
        "fold_id": fold,
        "site": site,
    }
    for row in payload["nested_selected"]["decisions"]:
        mask = (fold == int(row["held_fold"])) & (site == int(row["site"]))
        candidate = int(np.flatnonzero(names == str(row["candidate"]))[0])
        output["candidate_index"][mask] = candidate
        output["fixed_scale"][mask] = float(row["scale"])
        output["predicted_energy"][mask] = metrics["predicted_energy"][candidate, mask]
        output["truth_energy"][mask] = metrics["truth_energy"][candidate, mask]
        output["cross"][mask] = metrics["cross"][candidate, mask]
    output["candidate_names"] = names
    return output


def weight_features(weights: np.ndarray, prefix: str) -> tuple[list[np.ndarray], list[str]]:
    absolute = np.abs(weights).astype(np.float64)
    total = np.maximum(np.sum(absolute, axis=1), TINY)
    probability = absolute / total[:, None]
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-30)), axis=1)
    rows = [
        np.sum(weights, axis=1),
        np.sum(absolute, axis=1),
        np.sqrt(np.sum(weights * weights, axis=1)),
        np.max(weights, axis=1),
        np.min(weights, axis=1),
        np.sum(np.maximum(weights, 0.0), axis=1),
        np.sum(np.maximum(-weights, 0.0), axis=1),
        1.0 / np.maximum(np.sum(probability * probability, axis=1), TINY),
        entropy,
    ]
    names = [
        f"{prefix}_sum",
        f"{prefix}_l1",
        f"{prefix}_l2",
        f"{prefix}_max",
        f"{prefix}_min",
        f"{prefix}_positive_mass",
        f"{prefix}_negative_mass",
        f"{prefix}_effective_donors",
        f"{prefix}_entropy",
    ]
    return rows, names


def path_confidence_features(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    donor_ids: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    cache: ComponentCache,
    config: Config,
) -> tuple[np.ndarray, list[str]]:
    beta_table = load_beta_table()
    transport_config = TransportConfig(
        components=24,
        transported_components=config.paths,
        oversample=2,
        refinement_sweeps=2,
        donor_count=config.path_donors,
        direction_bins=3600,
        beta=140.40,
        radius_fractions=(1.0,),
    )
    names = [
        "path_match_coverage",
        "path_mean_observations",
        "path_component_coherence",
        "path_raw_phase_concentration",
        "path_radial_phase_concentration",
        "path_horizontal_delta",
        "path_vertical_delta",
        "path_delay_delta",
        "path_delay_length_delta_m",
    ]
    output = np.zeros((len(query_ids), len(names)), dtype=np.float32)
    for row, query_id in enumerate(query_ids):
        selected = donor_ids[row, : config.path_donors].astype(np.int64)
        anchor = int(selected[0])
        anchor_coordinates = cache.coordinates[anchor, : config.paths]
        anchor_coefficients = cache.coefficients[anchor, : config.paths]
        anchor_energy = component_energy(anchor_coefficients).astype(np.float64)
        anchor_weight = anchor_energy / max(float(np.sum(anchor_energy)), TINY)
        observations: list[list[tuple[int, int, float]]] = [
            [(anchor, path, 1.0)] for path in range(config.paths)
        ]
        for donor in selected[1:]:
            donor = int(donor)
            matched, _, coherence = match_components(
                anchor_coordinates,
                anchor_coefficients,
                cache.coordinates[donor],
                cache.coefficients[donor],
                transport_config,
            )
            for path, component in enumerate(matched):
                if component >= 0:
                    observations[path].append(
                        (donor, int(component), float(coherence[path]))
                    )
        beta = beta_table[(int(fold_id[row]), int(site[query_id]))]
        radial = np.linalg.norm(
            positions - bs_positions[int(site[query_id])][None], axis=1
        )
        per_path = np.zeros((config.paths, len(names)), dtype=np.float64)
        for path, values in enumerate(observations):
            ids = np.asarray([value[0] for value in values], dtype=np.int64)
            components = np.asarray([value[1] for value in values], dtype=np.int64)
            match_coherence = np.asarray([value[2] for value in values], dtype=np.float64)
            coordinates = cache.coordinates[ids, components]
            coefficients = cache.coefficients[ids, components]
            norm = np.linalg.norm(coefficients, axis=1)
            valid = norm > 1e-20
            if int(np.sum(valid)) < 2:
                continue
            ids = ids[valid]
            coordinates = coordinates[valid]
            coefficients = coefficients[valid]
            norm = norm[valid]
            match_coherence = match_coherence[valid]
            unit = coefficients / norm[:, None]
            inner = np.sum(unit[0][None] * np.conj(unit), axis=1)
            phase = np.angle(inner)
            weight = np.maximum(match_coherence, 0.10)
            weight /= max(float(np.sum(weight)), TINY)
            raw_concentration = float(np.abs(np.sum(weight * np.exp(1j * phase))))
            radial_delta = radial[ids] - radial[ids[0]]
            radial_concentration = float(
                np.abs(np.sum(weight * np.exp(1j * (phase - beta * radial_delta))))
            )
            delta_h = np.abs(
                circular_delta(coordinates[:, 0], coordinates[0, 0], MH)
            )
            delta_v = np.abs(
                circular_delta(coordinates[:, 1], coordinates[0, 1], MV)
            )
            delta_d = np.abs(
                circular_delta(coordinates[:, 2], coordinates[0, 2], S_SC)
            )
            per_path[path] = [
                min(len(ids) / config.path_donors, 1.0),
                float(len(ids)),
                float(np.sum(weight * match_coherence)),
                raw_concentration,
                radial_concentration,
                float(np.sum(weight * delta_h)),
                float(np.sum(weight * delta_v)),
                float(np.sum(weight * delta_d)),
                float(np.sum(weight * delta_d) * METERS_PER_DELAY_BIN),
            ]
        output[row] = (anchor_weight[:, None] * per_path).sum(axis=0).astype(np.float32)
        if (row + 1) % 100 == 0:
            print(f"phase confidence path features {row + 1}/{len(query_ids)}", flush=True)
    return output, names


def build_features(config: Config) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    if FEATURE_CACHE.is_file():
        cached = np.load(FEATURE_CACHE)
        return (
            cached["features"].astype(np.float32),
            cached["feature_names"].tolist(),
            cached["query_ids"].astype(np.int64),
            cached["fold_id"].astype(np.int8),
        )

    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    setup, positions, _, site, _ = load_positions()
    geometry_cache = np.load(GEOMETRY)
    geometry = geometry_cache["train"][query_ids].astype(np.float32)
    geometry_names = [f"geometry_{name}" for name in geometry_cache["names"].tolist()]
    weights = np.load(OOF_WEIGHTS)
    donor_ids = weights["donor_ids"].astype(np.int64)

    relative = positions[query_ids] - setup.bs_positions[site[query_ids]]
    distance = np.linalg.norm(relative, axis=1)
    angle = np.arctan2(relative[:, 1], relative[:, 0])
    feature_rows: list[np.ndarray] = [
        positions[query_ids, 0],
        positions[query_ids, 1],
        relative[:, 0],
        relative[:, 1],
        relative[:, 2],
        distance,
        np.sin(angle),
        np.cos(angle),
        (site[query_ids] == 0).astype(np.float64),
        (site[query_ids] == 1).astype(np.float64),
    ]
    feature_names = [
        "x",
        "y",
        "bs_dx",
        "bs_dy",
        "bs_dz",
        "bs_distance",
        "bs_azimuth_sin",
        "bs_azimuth_cos",
        "site0",
        "site1",
    ]
    feature_rows.extend([geometry[:, index] for index in range(geometry.shape[1])])
    feature_names.extend(geometry_names)

    donor_distance = np.linalg.norm(
        positions[donor_ids, :2] - positions[query_ids, None, :2], axis=2
    )
    for quantile in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0):
        feature_rows.append(np.quantile(donor_distance, quantile, axis=1))
        feature_names.append(f"donor_distance_q{int(100 * quantile):03d}")
    for key in ("base_weights", "pas_weights", "pdp_weights"):
        rows, names = weight_features(weights[key].astype(np.float64), key)
        feature_rows.extend(rows)
        feature_names.extend(names)
    feature_rows.extend(
        [
            np.sum(np.abs(weights["pas_weights"] - weights["base_weights"]), axis=1),
            np.sum(np.abs(weights["pdp_weights"] - weights["base_weights"]), axis=1),
            np.sum(np.abs(weights["pas_weights"] - weights["pdp_weights"]), axis=1),
        ]
    )
    feature_names.extend(
        ["pas_base_weight_l1", "pdp_base_weight_l1", "pas_pdp_weight_l1"]
    )

    descriptors = np.load(DESCRIPTORS)
    global_pas = descriptors["global_pas"].astype(np.float64)
    global_pdp = descriptors["global_pdp"].astype(np.float64)
    for prefix, spectrum in (("pas", global_pas), ("pdp", global_pdp)):
        power = spectrum * spectrum
        total = np.maximum(np.sum(power, axis=1), TINY)
        sorted_power = np.sort(power, axis=1)[:, ::-1]
        for count in (1, 2, 4, 8, 16, 32):
            feature_rows.append(np.sum(sorted_power[:, :count], axis=1) / total)
            feature_names.append(f"{prefix}_top{count}_fraction")
    feature_rows.extend(
        [
            np.log10(np.maximum(descriptors["total_power"], 1e-30)),
            descriptors["pas_entropy"],
            descriptors["pdp_entropy"],
        ]
    )
    feature_names.extend(["log10_foundation_power", "pas_entropy", "pdp_entropy"])

    cache = load_components()
    path_features, path_names = path_confidence_features(
        query_ids,
        fold_id,
        donor_ids,
        positions,
        site,
        setup.bs_positions,
        cache,
        config,
    )
    feature_rows.extend([path_features[:, index] for index in range(path_features.shape[1])])
    feature_names.extend(path_names)

    selected = selected_v4_arrays()
    feature_rows.extend(
        [
            np.log10(np.maximum(selected["predicted_energy"], 1e-30)),
            selected["fixed_scale"],
        ]
    )
    feature_names.extend(["log10_v4_unscaled_power", "v4_fixed_scale"])

    features = np.column_stack(feature_rows).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURE_CACHE,
        features=features,
        feature_names=np.asarray(feature_names),
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
    )
    return features, feature_names, query_ids, fold_id


def make_model(name: str, config: Config, seed: int):
    if name == "ridge10":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "ridge100":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=config.extra_trees,
            max_depth=config.extra_depth,
            min_samples_leaf=config.extra_leaf,
            max_features=0.70,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hist_gradient":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=10.0,
            random_state=seed,
        )
    raise ValueError(name)


def fit_model(model, features: np.ndarray, target: np.ndarray, weight: np.ndarray) -> None:
    if hasattr(model, "named_steps"):
        model.fit(features, target, ridge__sample_weight=weight)
    else:
        model.fit(features, target, sample_weight=weight)


def shuffled_target(
    target: np.ndarray,
    fit_rows: np.ndarray,
    site: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.array(target[fit_rows], copy=True)
    fit_site = site[fit_rows]
    for value in np.unique(fit_site):
        rows = np.flatnonzero(fit_site == value)
        output[rows] = output[rng.permutation(rows)]
    return output


def cross_validated_predictions(
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    config: Config,
) -> dict[str, np.ndarray]:
    model_names = ("site_mean", "ridge10", "ridge100", "extra_trees", "hist_gradient")
    predictions = {name: np.zeros(len(target), dtype=np.float64) for name in model_names}
    predictions["extra_trees_shuffle"] = np.zeros(len(target), dtype=np.float64)
    for held_fold in np.unique(fold_id):
        fit_rows = np.flatnonzero(fold_id != held_fold)
        held_rows = np.flatnonzero(fold_id == held_fold)
        for value in np.unique(site):
            fit_site = fit_rows[site[fit_rows] == value]
            held_site = held_rows[site[held_rows] == value]
            mean = float(
                np.sum(weight[fit_site] * target[fit_site])
                / max(float(np.sum(weight[fit_site])), TINY)
            )
            predictions["site_mean"][held_site] = mean
        for model_name in model_names[1:]:
            model = make_model(model_name, config, config.seed + int(held_fold) * 101)
            fit_model(model, features[fit_rows], target[fit_rows], weight[fit_rows])
            predictions[model_name][held_rows] = model.predict(features[held_rows])
        rng = np.random.default_rng(config.seed + int(held_fold) * 1009)
        control_target = shuffled_target(target, fit_rows, site, rng)
        control = make_model("extra_trees", config, config.seed + int(held_fold) * 101 + 7)
        fit_model(control, features[fit_rows], control_target, weight[fit_rows])
        predictions["extra_trees_shuffle"][held_rows] = control.predict(features[held_rows])
        print(f"phase confidence scaling fold={int(held_fold)}", flush=True)
    return {
        name: np.clip(values, 0.0, config.maximum_scale)
        for name, values in predictions.items()
    }


def nmse_for_scale(
    scale: np.ndarray,
    predicted_energy: np.ndarray,
    truth_energy: np.ndarray,
    cross: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    if mask is None:
        mask = np.ones(len(scale), dtype=bool)
    error = (
        scale[mask] * scale[mask] * predicted_energy[mask]
        + truth_energy[mask]
        - 2.0 * scale[mask] * cross[mask]
    )
    return float(np.sum(np.maximum(error, 0.0)) / max(float(np.sum(truth_energy[mask])), TINY))


def evaluate_predictions(
    predictions: dict[str, np.ndarray],
    selected: dict[str, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, object]:
    fixed = selected["fixed_scale"]
    pe = selected["predicted_energy"]
    te = selected["truth_energy"]
    cross = selected["cross"]
    baseline_nmse = nmse_for_scale(fixed, pe, te, cross)
    variants: dict[str, object] = {
        "v4_fixed_scale": {
            "nmse": baseline_nmse,
            "nmse_score": float(1.0 / (1.0 + baseline_nmse)),
            "total_score_gain_vs_v4": 0.0,
            "fold_nmse": [
                nmse_for_scale(fixed, pe, te, cross, fold_id == fold)
                for fold in np.unique(fold_id)
            ],
        }
    }
    all_scales = {"v4_fixed_scale": fixed}
    for name, prediction in predictions.items():
        for blend in (0.25, 0.50, 1.00):
            variant_name = f"{name}_blend{blend:.2f}"
            scale = (1.0 - blend) * fixed + blend * prediction
            all_scales[variant_name] = scale
            nmse = nmse_for_scale(scale, pe, te, cross)
            fold_nmse = [
                nmse_for_scale(scale, pe, te, cross, fold_id == fold)
                for fold in np.unique(fold_id)
            ]
            variants[variant_name] = {
                "nmse": nmse,
                "nmse_score": float(1.0 / (1.0 + nmse)),
                "total_score_gain_vs_v4": float(
                    0.2 * (1.0 / (1.0 + nmse) - 1.0 / (1.0 + baseline_nmse))
                ),
                "positive_fold_count": int(
                    np.sum(np.asarray(fold_nmse) < np.asarray(variants["v4_fixed_scale"]["fold_nmse"]))
                ),
                "fold_nmse": fold_nmse,
                "scale": summarize(scale),
            }
    oracle = np.clip(
        cross / np.maximum(pe, TINY), 0.0, Config().maximum_scale
    )
    oracle_nmse = nmse_for_scale(oracle, pe, te, cross)
    ranking = sorted(variants, key=lambda name: float(variants[name]["nmse"]))
    return {
        "baseline_nmse": baseline_nmse,
        "clipped_point_oracle": {
            "nmse": oracle_nmse,
            "total_score_gain_vs_v4": float(
                0.2
                * (
                    1.0 / (1.0 + oracle_nmse)
                    - 1.0 / (1.0 + baseline_nmse)
                )
            ),
            "scale": summarize(oracle),
        },
        "variants": variants,
        "ranking_low_nmse_first": ranking,
        "scales": all_scales,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.recompute_features and FEATURE_CACHE.is_file():
        FEATURE_CACHE.unlink()
    config = Config()
    features, feature_names, query_ids, fold_id = build_features(config)
    selected = selected_v4_arrays()
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids]
    target = np.clip(
        selected["cross"] / np.maximum(selected["predicted_energy"], TINY),
        0.0,
        config.maximum_scale,
    )
    predictions = cross_validated_predictions(
        features,
        target,
        selected["predicted_energy"],
        fold_id,
        site,
        config,
    )
    evaluation = evaluate_predictions(predictions, selected, fold_id)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PREDICTION_CACHE,
        **predictions,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        feature_names=np.asarray(feature_names),
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "Observable donor/path phase coherence predicts when V4's complex carrier is useful; "
            "pointwise nonnegative shrinkage can lower NMSE without changing normalized PAS/PDP."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "validation": (
            "fixed five complete spatial holes; each scale model trains only on the other four "
            "OOF folds; the phase-label shuffle keeps model capacity and site composition fixed"
        ),
        "features": {
            "count": int(features.shape[1]),
            "names": feature_names,
        },
        "evaluation": {key: value for key, value in evaluation.items() if key != "scales"},
        "outputs": {
            "features": str(FEATURE_CACHE.relative_to(PATHS.root)),
            "predictions": str(PREDICTION_CACHE.relative_to(PATHS.root)),
        },
        "seconds": float(time.time() - started),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
