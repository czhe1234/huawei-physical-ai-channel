#!/usr/bin/env python3
"""Audit nested official calibration of the frozen v375 lifecycle embedding."""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v370_path_track_lifecycle_pdp as v370
import analyze_v371_track_posterior_pdp as v371
import analyze_v373_long_family_visibility as v373
import analyze_v376_frozen_lifecycle_transfer as v376
import lifecycle_geometry_features as lgf
import train_v375_synthetic_lifecycle_pretraining as v375
from project_paths import PATHS


VERSION_NUMBER = 377
VERSION = "v377_nested_embedding_calibration"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
PAIR_CACHE = OUT_DIR / "v377_nested_embedding_calibration_pairs.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
CHECKPOINT = PATHS.cache_dir(375) / "v375_synthetic_lifecycle_model.pt"

EXISTENCE_L2 = 0.10
MASS_RIDGE = 100.0
MAX_OPTIMIZER_ITERATIONS = 300

MIN_FULL_VS_LOCAL_AP = 0.002
MIN_FULL_VS_LOCAL_MASS = 0.002
MIN_FULL_VS_SHUFFLED_AP = 0.002
MIN_FULL_VS_SHUFFLED_MASS = 0.002
MIN_FULL_VS_RAW_AP = 0.001
MIN_FULL_VS_RAW_MASS = 0.001
MIN_FOLD_WINS = 4


def frozen_embedding(
    model: v375.LifecycleNet,
    feature: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    normalized = np.clip((feature - mean) / std, -8.0, 8.0).astype(np.float32)
    output = np.empty((len(feature), 128), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(feature), 16384):
            end = min(start + 16384, len(feature))
            batch = torch.from_numpy(normalized[start:end]).to(device)
            output[start:end] = model.trunk(batch).cpu().numpy()
    return output


def representation_sets(
    *,
    model: v375.LifecycleNet,
    feature: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    shuffled = v375.shuffled_geometry(
        feature, np.zeros(len(feature), dtype=np.int8), seed
    )
    normalized_raw = np.clip((feature - mean) / std, -8.0, 8.0).astype(np.float32)
    return {
        "identity_only": np.empty((len(feature), 0), dtype=np.float32),
        "raw_geometry": normalized_raw,
        "shuffled_embedding": frozen_embedding(
            model, shuffled, mean, std, device
        ),
        "frozen_embedding": frozen_embedding(
            model, feature, mean, std, device
        ),
    }


def design_matrix(identity: np.ndarray, representation: np.ndarray) -> np.ndarray:
    identity_feature = v376.logit(identity).reshape(-1, 1).astype(np.float32)
    return np.concatenate([identity_feature, representation], axis=1)


def standardize(
    train: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, dtype=np.float64)
    std = np.std(train, axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-3)
    train_scaled = np.clip((train - mean) / std, -8.0, 8.0).astype(np.float64)
    query_scaled = np.clip((query - mean) / std, -8.0, 8.0).astype(np.float64)
    train_scaled = np.column_stack([np.ones(len(train_scaled)), train_scaled])
    query_scaled = np.column_stack([np.ones(len(query_scaled)), query_scaled])
    return train_scaled, query_scaled, mean.astype(np.float32), std.astype(np.float32)


def fit_offset_logistic(
    design: np.ndarray,
    offset: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    target = target.astype(np.float64)
    offset = np.clip(offset.astype(np.float64), -12.0, 12.0)
    dimension = design.shape[1]

    def objective(weight: np.ndarray) -> tuple[float, np.ndarray]:
        score = np.clip(offset + design @ weight, -30.0, 30.0)
        probability = v376.sigmoid(score)
        loss = np.mean(np.logaddexp(0.0, score) - target * score)
        regularized = weight.copy()
        regularized[0] = 0.0
        loss += 0.5 * EXISTENCE_L2 * float(np.dot(regularized, regularized))
        gradient = design.T @ (probability - target) / len(target)
        gradient += EXISTENCE_L2 * regularized
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": MAX_OPTIMIZER_ITERATIONS, "ftol": 1e-10},
    )
    return result.x.astype(np.float64), {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
    }


def fit_calibrator(
    *,
    train_design: np.ndarray,
    train_local_probability: np.ndarray,
    train_local_mass: np.ndarray,
    train_binary: np.ndarray,
    train_mass: np.ndarray,
    query_design: np.ndarray,
    query_local_probability: np.ndarray,
    query_local_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_scaled, query_scaled, mean, std = standardize(
        train_design, query_design
    )
    train_logit = v376.logit(train_local_probability)
    query_logit = v376.logit(query_local_probability)
    existence_weight, optimizer = fit_offset_logistic(
        train_scaled, train_logit, train_binary
    )
    query_probability = v376.sigmoid(
        query_logit + query_scaled @ existence_weight
    ).astype(np.float32)

    train_conditional = np.divide(
        train_local_mass, np.maximum(train_local_probability, 1e-5)
    )
    query_conditional = np.divide(
        query_local_mass, np.maximum(query_local_probability, 1e-5)
    )
    positive = train_binary.astype(bool) & (train_mass > 0.0)
    if int(np.sum(positive)) < 20:
        raise RuntimeError("too few positive lifecycle calibration examples")
    residual = (
        np.log(np.maximum(train_mass[positive], 1e-8))
        - np.log(np.maximum(train_conditional[positive], 1e-8))
    )
    mass_model = Ridge(alpha=MASS_RIDGE, fit_intercept=False)
    mass_model.fit(train_scaled[positive], residual)
    query_log_conditional = (
        np.log(np.maximum(query_conditional, 1e-8))
        + mass_model.predict(query_scaled)
    )
    query_mass = (
        query_probability * np.exp(np.clip(query_log_conditional, -16.0, 2.0))
    ).astype(np.float32)
    diagnostics: dict[str, object] = {
        "feature_dimension": int(train_design.shape[1]),
        "train_rows": int(len(train_design)),
        "train_positive": int(np.sum(positive)),
        "optimizer": optimizer,
        "existence_weight_l2": float(np.linalg.norm(existence_weight[1:])),
        "mass_weight_l2": float(np.linalg.norm(mass_model.coef_[1:])),
        "standardization_mean_abs": float(np.mean(np.abs(mean))),
        "standardization_std_mean": float(np.mean(std)),
    }
    return query_probability, query_mass, diagnostics


def scatter_calibrated_scores(
    exposure: v373.Exposures,
    probability: np.ndarray,
    mass: np.ndarray,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_probability = np.zeros(node_count, dtype=np.float32)
    output_mass = np.zeros(node_count, dtype=np.float32)
    covered = np.zeros(node_count, dtype=bool)
    for row, node in enumerate(exposure.node):
        if not covered[node] or probability[row] > output_probability[node]:
            covered[node] = True
            output_probability[node] = probability[row]
            output_mass[node] = mass[row]
    return output_probability, output_mass, covered


def apply_calibrated_exposures(
    *,
    exposure: v373.Exposures,
    probability: np.ndarray,
    mass: np.ndarray,
    local_probability: np.ndarray,
    local_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scattered_probability, scattered_mass, covered = scatter_calibrated_scores(
        exposure, probability, mass, len(local_probability)
    )
    output_probability = local_probability.copy()
    output_mass = local_mass.copy()
    output_probability[covered] = scattered_probability[covered]
    output_mass[covered] = scattered_mass[covered]
    return output_probability, output_mass, covered


def build_fold_examples(
    *,
    fit_ids: np.ndarray,
    held_ids: np.ndarray,
    valid_ids: np.ndarray,
    geometry: tuple[np.ndarray, ...],
    node_raw: np.ndarray,
    component: dict[str, np.ndarray],
    all_pos: np.ndarray,
    point_feature: np.ndarray,
    raster: lgf.RasterScene,
    bs: np.ndarray,
    model: v375.LifecycleNet,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    bundle = v370.prepare_bundle(valid_ids, fit_ids, geometry, node_raw, component)
    families, discovery = v373.discover_families(
        bundle, fit_ids, all_pos[:, :2]
    )
    if not families:
        raise RuntimeError("nested fold discovered no lifecycle families")
    exposure = v373.build_exposures(
        families, bundle, held_ids, all_pos[:, :2]
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
    representations = representation_sets(
        model=model,
        feature=feature,
        mean=feature_mean,
        std=feature_std,
        device=device,
        seed=seed,
    )
    local_probability, local_mass, _ = v371.fit_local_scores(
        bundle, fit_ids, held_ids, seed + 1000
    )
    return {
        "bundle": bundle,
        "families": families,
        "discovery": discovery,
        "exposure": exposure,
        "feature": feature,
        "representations": representations,
        "local_probability": local_probability,
        "local_mass": local_mass,
    }


def concatenate_training_examples(
    rows: list[dict[str, object]], variants: tuple[str, ...]
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    representation: dict[str, np.ndarray] = {}
    for variant in variants:
        representation[variant] = np.concatenate(
            [row["representations"][variant] for row in rows]  # type: ignore[index]
        )
    return {
        "identity": np.concatenate(
            [row["exposure"].identity for row in rows]  # type: ignore[union-attr]
        ),
        "binary": np.concatenate(
            [row["exposure"].binary for row in rows]  # type: ignore[union-attr]
        ),
        "mass": np.concatenate(
            [row["exposure"].mass for row in rows]  # type: ignore[union-attr]
        ),
        "local_probability": np.concatenate(
            [
                row["local_probability"][row["exposure"].node]  # type: ignore[index,union-attr]
                for row in rows
            ]
        ),
        "local_mass": np.concatenate(
            [
                row["local_mass"][row["exposure"].node]  # type: ignore[index,union-attr]
                for row in rows
            ]
        ),
        "representation": representation,
    }


def main() -> None:
    for path in (
        TARGET_CACHE,
        CHECKPOINT,
        v297.COMPONENT_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
        v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = v375.LifecycleNet(lgf.feature_dimensions()["total"])
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

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
    variants = (
        "identity_only",
        "raw_geometry",
        "shuffled_embedding",
        "frozen_embedding",
    )

    fold_rows: list[dict[str, object]] = []
    cache: dict[str, list[np.ndarray]] = {
        "fold": [],
        "truth_binary": [],
        "truth_mass": [],
        "local_probability": [],
        "local_mass": [],
        "covered": [],
    }
    for variant in variants:
        cache[f"{variant}_probability"] = []
        cache[f"{variant}_mass"] = []

    for outer_fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        nested_rows = []
        for inner_fold, inner_ids_all in enumerate(folds):
            if inner_fold == outer_fold:
                continue
            inner_held = np.intersect1d(fit_ids, inner_ids_all)
            inner_fit = np.setdiff1d(fit_ids, inner_held)
            nested_rows.append(
                build_fold_examples(
                    fit_ids=inner_fit,
                    held_ids=inner_held,
                    valid_ids=valid_ids,
                    geometry=geometry,
                    node_raw=node_raw,
                    component=component,
                    all_pos=all_pos,
                    point_feature=point_feature,
                    raster=raster,
                    bs=bs,
                    model=model,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    device=device,
                    seed=377_000 + outer_fold * 100 + inner_fold,
                )
            )
        train = concatenate_training_examples(nested_rows, variants)
        outer = build_fold_examples(
            fit_ids=fit_ids,
            held_ids=outer_ids,
            valid_ids=valid_ids,
            geometry=geometry,
            node_raw=node_raw,
            component=component,
            all_pos=all_pos,
            point_feature=point_feature,
            raster=raster,
            bs=bs,
            model=model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
            seed=377_500 + outer_fold,
        )
        outer_bundle = outer["bundle"]
        outer_exposure = outer["exposure"]
        outer_local_probability = outer["local_probability"]
        outer_local_mass = outer["local_mass"]
        local_metrics = v373.subset_metrics(
            outer_bundle,
            outer_ids,
            outer_local_probability,
            outer_local_mass,
            None,
        )

        variant_metrics: dict[str, dict[str, float]] = {}
        variant_diagnostics: dict[str, dict[str, object]] = {}
        variant_outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for variant in variants:
            train_design = design_matrix(
                train["identity"], train["representation"][variant]  # type: ignore[index]
            )
            query_design = design_matrix(
                outer_exposure.identity,
                outer["representations"][variant],  # type: ignore[index]
            )
            probability, mass, diagnostics = fit_calibrator(
                train_design=train_design,
                train_local_probability=train["local_probability"],  # type: ignore[arg-type]
                train_local_mass=train["local_mass"],  # type: ignore[arg-type]
                train_binary=train["binary"],  # type: ignore[arg-type]
                train_mass=train["mass"],  # type: ignore[arg-type]
                query_design=query_design,
                query_local_probability=outer_local_probability[outer_exposure.node],
                query_local_mass=outer_local_mass[outer_exposure.node],
            )
            output = apply_calibrated_exposures(
                exposure=outer_exposure,
                probability=probability,
                mass=mass,
                local_probability=outer_local_probability,
                local_mass=outer_local_mass,
            )
            variant_outputs[variant] = output
            variant_metrics[variant] = v373.subset_metrics(
                outer_bundle, outer_ids, output[0], output[1], None
            )
            variant_diagnostics[variant] = diagnostics

        full = variant_metrics["frozen_embedding"]
        shuffled = variant_metrics["shuffled_embedding"]
        raw = variant_metrics["raw_geometry"]
        deltas = {
            "full_vs_local_ap": float(
                full["existence_ap"] - local_metrics["existence_ap"]
            ),
            "full_vs_local_mass": float(
                full["mass_cosine"] - local_metrics["mass_cosine"]
            ),
            "full_vs_shuffled_ap": float(
                full["existence_ap"] - shuffled["existence_ap"]
            ),
            "full_vs_shuffled_mass": float(
                full["mass_cosine"] - shuffled["mass_cosine"]
            ),
            "full_vs_raw_ap": float(
                full["existence_ap"] - raw["existence_ap"]
            ),
            "full_vs_raw_mass": float(
                full["mass_cosine"] - raw["mass_cosine"]
            ),
        }
        fold_rows.append(
            {
                "fold": outer_fold,
                "outer_query_count": int(len(outer_ids)),
                "nested_training_exposures": int(len(train["binary"])),
                "outer_exposures": int(len(outer_exposure.node)),
                "local": local_metrics,
                "variants": variant_metrics,
                "calibration": variant_diagnostics,
                "deltas": deltas,
                "leakage_assertions": {
                    "outer_in_nested_training": 0,
                    "outer_h_in_family_discovery": 0,
                    "outer_h_in_embedding": 0,
                    "outer_h_in_local_model": 0,
                    "synthetic_official_target": 0,
                },
            }
        )

        outer_nodes = np.isin(outer_bundle.node_query_id, outer_ids)
        cache["fold"].append(
            np.full(np.sum(outer_nodes), outer_fold, dtype=np.int8)
        )
        cache["truth_binary"].append(
            outer_bundle.arrays["binary"][outer_nodes].astype(np.int8)
        )
        cache["truth_mass"].append(outer_bundle.arrays["mass"][outer_nodes])
        cache["local_probability"].append(outer_local_probability[outer_nodes])
        cache["local_mass"].append(outer_local_mass[outer_nodes])
        cache["covered"].append(
            variant_outputs["frozen_embedding"][2][outer_nodes]
        )
        for variant in variants:
            cache[f"{variant}_probability"].append(
                variant_outputs[variant][0][outer_nodes]
            )
            cache[f"{variant}_mass"].append(
                variant_outputs[variant][1][outer_nodes]
            )
        print(
            f"v377 fold={outer_fold} train={len(train['binary'])} "
            f"outer={len(outer_exposure.node)} "
            f"delta_local={deltas['full_vs_local_ap']:+.4f}/"
            f"{deltas['full_vs_local_mass']:+.4f} "
            f"delta_shuffle={deltas['full_vs_shuffled_ap']:+.4f}/"
            f"{deltas['full_vs_shuffled_mass']:+.4f}",
            flush=True,
        )

    merged = {key: np.concatenate(value) for key, value in cache.items()}
    np.savez_compressed(PAIR_CACHE, **merged)
    mean = {
        key: float(np.mean([row["deltas"][key] for row in fold_rows]))
        for key in fold_rows[0]["deltas"]
    }
    fold_wins = int(
        sum(
            row["deltas"]["full_vs_local_ap"] > 0.0
            and row["deltas"]["full_vs_local_mass"] > 0.0
            and row["deltas"]["full_vs_shuffled_ap"] > 0.0
            and row["deltas"]["full_vs_shuffled_mass"] > 0.0
            for row in fold_rows
        )
    )
    passed = bool(
        mean["full_vs_local_ap"] >= MIN_FULL_VS_LOCAL_AP
        and mean["full_vs_local_mass"] >= MIN_FULL_VS_LOCAL_MASS
        and mean["full_vs_shuffled_ap"] >= MIN_FULL_VS_SHUFFLED_AP
        and mean["full_vs_shuffled_mass"] >= MIN_FULL_VS_SHUFFLED_MASS
        and mean["full_vs_raw_ap"] >= MIN_FULL_VS_RAW_AP
        and mean["full_vs_raw_mass"] >= MIN_FULL_VS_RAW_MASS
        and fold_wins >= MIN_FOLD_WINS
    )
    summary = {**mean, "fold_wins": fold_wins}
    report = {
        "method": VERSION,
        "status": "embedding_transfer_passed" if passed else "analysis_only_no_transfer",
        "scope": "strict nested official path-level audit; no PDP/full-H/test target/ZIP",
        "primary_hypothesis": (
            "The frozen v375 penultimate representation contains official lifecycle "
            "information complementary to the fit-only local model after a small "
            "official crossfit calibration head removes posterior domain shift."
        ),
        "parameters": {
            "checkpoint": PATHS.relative(CHECKPOINT),
            "existence_l2": EXISTENCE_L2,
            "mass_ridge": MASS_RIDGE,
            "optimizer_iterations": MAX_OPTIMIZER_ITERATIONS,
            "controls": list(variants[:-1]),
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "full_vs_local_ap": f">={MIN_FULL_VS_LOCAL_AP}",
            "full_vs_local_mass": f">={MIN_FULL_VS_LOCAL_MASS}",
            "full_vs_shuffled_ap": f">={MIN_FULL_VS_SHUFFLED_AP}",
            "full_vs_shuffled_mass": f">={MIN_FULL_VS_SHUFFLED_MASS}",
            "full_vs_raw_ap": f">={MIN_FULL_VS_RAW_AP}",
            "full_vs_raw_mass": f">={MIN_FULL_VS_RAW_MASS}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "passed": passed,
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "pair_cache": PATHS.relative(PAIR_CACHE),
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
