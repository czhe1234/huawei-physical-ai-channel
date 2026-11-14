#!/usr/bin/env python3
"""Strictly calibrate v386 residual strengths with nested official folds."""

from __future__ import annotations

import json

import numpy as np
import torch

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v370_path_track_lifecycle_pdp as v370
import analyze_v371_track_posterior_pdp as v371
import analyze_v373_long_family_visibility as v373
import analyze_v376_frozen_lifecycle_transfer as v376
import analyze_v386_dense_residual_official_transfer as v386
import lifecycle_geometry_features as lgf
import train_v375_synthetic_lifecycle_pretraining as v375
import train_v384_dense_boundary_residual as v384
from project_paths import PATHS


VERSION_NUMBER = 387
VERSION = "v387_nested_residual_calibration"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
PAIR_CACHE = OUT_DIR / "v387_nested_residual_calibration_pairs.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
CHECKPOINT = PATHS.cache_dir(384) / "v384_dense_boundary_residual_model.pt"
V385_REPORT = PATHS.version_report_dir(385) / "v385_residual_head_attribution.json"

EXISTENCE_STRENGTHS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35)
MASS_STRENGTHS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)

MIN_FULL_VS_LOCAL_AP = 0.002
MIN_FULL_VS_LOCAL_MASS = 0.002
MIN_FULL_VS_SHUFFLED_AP = 0.001
MIN_FULL_VS_SHUFFLED_MASS = 0.001
MIN_FOLD_WINS = 4


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-4, 1.0 - 1e-4)
    return np.log(probability) - np.log1p(-probability)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def apply_strength(
    *,
    local_probability: np.ndarray,
    local_mass: np.ndarray,
    raw_probability: np.ndarray,
    raw_mass: np.ndarray,
    covered: np.ndarray,
    existence_strength: float,
    mass_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    probability = local_probability.copy()
    mass = local_mass.copy()
    local_conditional = np.divide(
        local_mass[covered], np.maximum(local_probability[covered], 1e-5)
    )
    raw_conditional = np.divide(
        raw_mass[covered], np.maximum(raw_probability[covered], 1e-5)
    )
    probability[covered] = sigmoid(
        logit(local_probability[covered])
        + existence_strength
        * (logit(raw_probability[covered]) - logit(local_probability[covered]))
    ).astype(np.float32)
    conditional = np.exp(
        np.log(np.maximum(local_conditional, 1e-8))
        + mass_strength
        * (
            np.log(np.maximum(raw_conditional, 1e-8))
            - np.log(np.maximum(local_conditional, 1e-8))
        )
    )
    mass[covered] = probability[covered] * conditional
    return probability, mass.astype(np.float32)


def raw_fold_outputs(
    *,
    valid_ids: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    geometry: tuple[np.ndarray, ...],
    node_raw: np.ndarray,
    component: dict[str, np.ndarray],
    all_pos: np.ndarray,
    point_feature: np.ndarray,
    raster: lgf.RasterScene,
    bs: np.ndarray,
    model: v384.BoundaryResidualNet,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    synthetic_mass_strength: float,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    bundle = v370.prepare_bundle(valid_ids, fit_ids, geometry, node_raw, component)
    families, discovery = v373.discover_families(
        bundle, fit_ids, all_pos[:, :2]
    )
    exposure = v373.build_exposures(
        families, bundle, query_ids, all_pos[:, :2]
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
    shuffled_feature = v375.shuffled_geometry(
        feature, np.zeros(len(feature), dtype=np.int8), seed
    )
    local_probability, local_mass, _ = v371.fit_local_scores(
        bundle, fit_ids, query_ids, seed + 1000
    )
    pair_local_probability = local_probability[exposure.node]
    pair_local_mass = local_mass[exposure.node]
    full_pair_probability, full_pair_mass, _ = v386.pair_prediction(
        model=model,
        feature=feature,
        feature_mean=feature_mean,
        feature_std=feature_std,
        local_probability=pair_local_probability,
        local_mass=pair_local_mass,
        mass_strength=synthetic_mass_strength,
        device=device,
    )
    shuffled_pair_probability, shuffled_pair_mass, _ = v386.pair_prediction(
        model=model,
        feature=shuffled_feature,
        feature_mean=feature_mean,
        feature_std=feature_std,
        local_probability=pair_local_probability,
        local_mass=pair_local_mass,
        mass_strength=synthetic_mass_strength,
        device=device,
    )
    full_probability, full_mass, covered = v373.scatter_exposure_scores(
        exposure,
        full_pair_probability / np.maximum(exposure.identity, 1e-5),
        full_pair_mass / np.maximum(exposure.identity, 1e-5),
        len(bundle.node_query_id),
    )
    shuffled_probability, shuffled_mass, _ = v373.scatter_exposure_scores(
        exposure,
        shuffled_pair_probability / np.maximum(exposure.identity, 1e-5),
        shuffled_pair_mass / np.maximum(exposure.identity, 1e-5),
        len(bundle.node_query_id),
    )
    raw_full_probability = local_probability.copy()
    raw_full_mass = local_mass.copy()
    raw_shuffled_probability = local_probability.copy()
    raw_shuffled_mass = local_mass.copy()
    raw_full_probability[covered] = full_probability[covered]
    raw_full_mass[covered] = full_mass[covered]
    raw_shuffled_probability[covered] = shuffled_probability[covered]
    raw_shuffled_mass[covered] = shuffled_mass[covered]
    return {
        "bundle": bundle,
        "discovery": discovery,
        "exposure_count": int(len(exposure.node)),
        "local_probability": local_probability,
        "local_mass": local_mass,
        "raw_full_probability": raw_full_probability,
        "raw_full_mass": raw_full_mass,
        "raw_shuffled_probability": raw_shuffled_probability,
        "raw_shuffled_mass": raw_shuffled_mass,
        "covered": covered,
    }


def metrics(
    output: dict[str, object],
    query_ids: np.ndarray,
    probability: np.ndarray,
    mass: np.ndarray,
) -> dict[str, float]:
    return v373.subset_metrics(
        output["bundle"], query_ids, probability, mass, None
    )


def select_strength(
    inner_rows: list[tuple[dict[str, object], np.ndarray]],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    local_ap = float(
        np.mean(
            [
                metrics(
                    output,
                    query_ids,
                    output["local_probability"],
                    output["local_mass"],
                )["existence_ap"]
                for output, query_ids in inner_rows
            ]
        )
    )
    local_mass = float(
        np.mean(
            [
                metrics(
                    output,
                    query_ids,
                    output["local_probability"],
                    output["local_mass"],
                )["mass_cosine"]
                for output, query_ids in inner_rows
            ]
        )
    )
    rows = []
    for existence_strength in EXISTENCE_STRENGTHS:
        for mass_strength in MASS_STRENGTHS:
            fold_metrics = []
            for output, query_ids in inner_rows:
                probability, mass = apply_strength(
                    local_probability=output["local_probability"],
                    local_mass=output["local_mass"],
                    raw_probability=output["raw_full_probability"],
                    raw_mass=output["raw_full_mass"],
                    covered=output["covered"],
                    existence_strength=existence_strength,
                    mass_strength=mass_strength,
                )
                fold_metrics.append(metrics(output, query_ids, probability, mass))
            ap = float(np.mean([row["existence_ap"] for row in fold_metrics]))
            mass_cosine = float(np.mean([row["mass_cosine"] for row in fold_metrics]))
            rows.append(
                {
                    "existence_strength": float(existence_strength),
                    "mass_strength": float(mass_strength),
                    "existence_ap": ap,
                    "mass_cosine": mass_cosine,
                    "ap_delta": ap - local_ap,
                    "mass_delta": mass_cosine - local_mass,
                    "objective": 0.5 * ap + 0.5 * mass_cosine,
                }
            )
    eligible = [
        row
        for row in rows
        if row["ap_delta"] >= 0.0 and row["mass_delta"] >= 0.0
    ]
    if not eligible:
        selected = next(
            row
            for row in rows
            if row["existence_strength"] == 0.0 and row["mass_strength"] == 0.0
        )
    else:
        selected = max(
            eligible,
            key=lambda row: (
                row["objective"],
                -row["existence_strength"],
                -row["mass_strength"],
            ),
        )
    return selected, rows


def main() -> None:
    for path in (
        TARGET_CACHE,
        CHECKPOINT,
        V385_REPORT,
        v297.COMPONENT_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
        v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    v385_report = json.loads(V385_REPORT.read_text(encoding="utf-8"))
    synthetic_mass_strength = float(v385_report["selected_mass_strength"])
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = v384.BoundaryResidualNet(lgf.feature_dimensions()["total"])
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

    fold_rows: list[dict[str, object]] = []
    cache: dict[str, list[np.ndarray]] = {
        "fold": [],
        "truth_binary": [],
        "truth_mass": [],
        "covered": [],
        "local_probability": [],
        "local_mass": [],
        "shuffled_probability": [],
        "shuffled_mass": [],
        "full_probability": [],
        "full_mass": [],
    }
    for outer_fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        inner_rows = []
        for inner_fold, inner_ids_all in enumerate(folds):
            if inner_fold == outer_fold:
                continue
            inner_ids = np.intersect1d(fit_ids, inner_ids_all)
            inner_fit = np.setdiff1d(fit_ids, inner_ids)
            output = raw_fold_outputs(
                valid_ids=valid_ids,
                fit_ids=inner_fit,
                query_ids=inner_ids,
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
                synthetic_mass_strength=synthetic_mass_strength,
                device=device,
                seed=387_000 + outer_fold * 100 + inner_fold,
            )
            inner_rows.append((output, inner_ids))
        selected, selection_rows = select_strength(inner_rows)
        outer = raw_fold_outputs(
            valid_ids=valid_ids,
            fit_ids=fit_ids,
            query_ids=outer_ids,
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
            synthetic_mass_strength=synthetic_mass_strength,
            device=device,
            seed=387_500 + outer_fold,
        )
        full_probability, full_mass = apply_strength(
            local_probability=outer["local_probability"],
            local_mass=outer["local_mass"],
            raw_probability=outer["raw_full_probability"],
            raw_mass=outer["raw_full_mass"],
            covered=outer["covered"],
            existence_strength=selected["existence_strength"],
            mass_strength=selected["mass_strength"],
        )
        shuffled_probability, shuffled_mass = apply_strength(
            local_probability=outer["local_probability"],
            local_mass=outer["local_mass"],
            raw_probability=outer["raw_shuffled_probability"],
            raw_mass=outer["raw_shuffled_mass"],
            covered=outer["covered"],
            existence_strength=selected["existence_strength"],
            mass_strength=selected["mass_strength"],
        )
        local_metrics = metrics(
            outer,
            outer_ids,
            outer["local_probability"],
            outer["local_mass"],
        )
        full_metrics = metrics(outer, outer_ids, full_probability, full_mass)
        shuffled_metrics = metrics(
            outer, outer_ids, shuffled_probability, shuffled_mass
        )
        deltas = {
            "full_vs_local_ap": float(
                full_metrics["existence_ap"] - local_metrics["existence_ap"]
            ),
            "full_vs_local_mass": float(
                full_metrics["mass_cosine"] - local_metrics["mass_cosine"]
            ),
            "full_vs_shuffled_ap": float(
                full_metrics["existence_ap"] - shuffled_metrics["existence_ap"]
            ),
            "full_vs_shuffled_mass": float(
                full_metrics["mass_cosine"] - shuffled_metrics["mass_cosine"]
            ),
        }
        fold_rows.append(
            {
                "fold": outer_fold,
                "outer_query_count": int(len(outer_ids)),
                "outer_exposures": outer["exposure_count"],
                "selected": selected,
                "selection_rows": selection_rows,
                "local": local_metrics,
                "shuffled_geometry": shuffled_metrics,
                "full": full_metrics,
                "deltas": deltas,
                "leakage_assertions": {
                    "outer_in_inner_strength_selection": 0,
                    "outer_h_in_family_discovery": 0,
                    "outer_h_in_local_model": 0,
                    "outer_h_in_residual_input": 0,
                },
            }
        )
        outer_nodes = np.isin(outer["bundle"].node_query_id, outer_ids)
        cache["fold"].append(
            np.full(np.sum(outer_nodes), outer_fold, dtype=np.int8)
        )
        cache["truth_binary"].append(
            outer["bundle"].arrays["binary"][outer_nodes].astype(np.int8)
        )
        cache["truth_mass"].append(outer["bundle"].arrays["mass"][outer_nodes])
        cache["covered"].append(outer["covered"][outer_nodes])
        cache["local_probability"].append(
            outer["local_probability"][outer_nodes]
        )
        cache["local_mass"].append(outer["local_mass"][outer_nodes])
        cache["shuffled_probability"].append(shuffled_probability[outer_nodes])
        cache["shuffled_mass"].append(shuffled_mass[outer_nodes])
        cache["full_probability"].append(full_probability[outer_nodes])
        cache["full_mass"].append(full_mass[outer_nodes])
        print(
            f"v387 fold={outer_fold} selected="
            f"{selected['existence_strength']:.2f}/{selected['mass_strength']:.2f} "
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
        and fold_wins >= MIN_FOLD_WINS
    )
    summary = {**mean, "fold_wins": fold_wins}
    report = {
        "method": VERSION,
        "status": "nested_calibration_passed" if passed else "analysis_only_no_transfer",
        "scope": "strict nested scalar calibration of frozen v385 official residual",
        "primary_hypothesis": (
            "A small nested-selected fraction of the v386 geometry correction improves "
            "the stronger official local posterior and still beats shuffled geometry."
        ),
        "parameters": {
            "existence_strengths": list(EXISTENCE_STRENGTHS),
            "mass_strengths": list(MASS_STRENGTHS),
            "eligibility": "inner mean AP and mass must both be >= local",
            "synthetic_mass_strength": synthetic_mass_strength,
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "full_vs_local_ap": f">={MIN_FULL_VS_LOCAL_AP}",
            "full_vs_local_mass": f">={MIN_FULL_VS_LOCAL_MASS}",
            "full_vs_shuffled_ap": f">={MIN_FULL_VS_SHUFFLED_AP}",
            "full_vs_shuffled_mass": f">={MIN_FULL_VS_SHUFFLED_MASS}",
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
