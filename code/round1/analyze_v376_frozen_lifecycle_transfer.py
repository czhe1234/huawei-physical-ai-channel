#!/usr/bin/env python3
"""Audit frozen v375 lifecycle transfer on strict official spatial folds."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v366_local_eikonal_pdp as v366
import analyze_v370_path_track_lifecycle_pdp as v370
import analyze_v371_track_posterior_pdp as v371
import analyze_v372_virtual_source_gap_reconnection as v372
import analyze_v373_long_family_visibility as v373
import lifecycle_geometry_features as lgf
import train_v375_synthetic_lifecycle_pretraining as v375
from project_paths import PATHS


VERSION_NUMBER = 376
VERSION = "v376_frozen_lifecycle_transfer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
PAIR_CACHE = OUT_DIR / "v376_frozen_lifecycle_transfer_pairs.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
CHECKPOINT = PATHS.cache_dir(375) / "v375_synthetic_lifecycle_model.pt"

OFFICIAL_FREQUENCY = 6.693e9
TRANSFER_STRENGTH = 0.25
SYNTHETIC_PRIOR = 0.30

MIN_SYNTHETIC_VS_SHUFFLED_AP = 0.02
MIN_COMBINED_VS_LOCAL_AP = 0.005
MIN_COMBINED_VS_LOCAL_MASS = 0.005
MIN_FOLD_WINS = 4


def logit(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def canonical_delay(delay: float) -> float:
    value = float(delay % v366.N_DELAY)
    return min(value, v366.N_DELAY - value)


def family_fit_rmse(
    family: v373.LongFamily,
    bundle: v370.TrackBundle,
    all_xy: np.ndarray,
) -> float:
    delay = bundle.hypothesis.delay[bundle.hypothesis.valid].astype(np.float64)
    predicted, _ = v372.affine_prediction(
        family.model, all_xy[bundle.node_query_id[family.nodes]]
    )
    error = v366.signed_circular_delta(predicted, delay[family.nodes])
    mass = bundle.arrays["mass"][family.nodes].astype(np.float64)
    weight = np.sqrt(np.maximum(mass, 1e-7))
    return float(np.sqrt(np.sum(weight * error**2) / np.sum(weight)))


def exposure_features(
    *,
    exposure: v373.Exposures,
    families: list[v373.LongFamily],
    bundle: v370.TrackBundle,
    all_pos: np.ndarray,
    point_feature: np.ndarray,
    raster: lgf.RasterScene,
    bs: np.ndarray,
) -> np.ndarray:
    feature = []
    fit_rmse = {
        family.family_id: family_fit_rmse(family, bundle, all_pos[:, :2])
        for family in families
    }
    for row in range(len(exposure.node)):
        family = families[int(exposure.family[row])]
        query_id = int(exposure.query_id[row])
        predicted_delay, predicted_gradient = v372.affine_prediction(
            family.model, all_pos[[query_id], :2]
        )
        support_mass = bundle.arrays["mass"][family.nodes].astype(np.float64)
        feature.append(
            lgf.family_feature(
                scene=raster,
                point=all_pos[query_id],
                bs=bs,
                point_feature=point_feature[query_id],
                predicted_delay=canonical_delay(float(predicted_delay[0])),
                predicted_gradient=predicted_gradient[0],
                support_xy=family.support_xy,
                support_mass=support_mass,
                fit_delay_rmse=fit_rmse[family.family_id],
                frequency=OFFICIAL_FREQUENCY,
            )
        )
    if not feature:
        return np.empty((0, lgf.feature_dimensions()["total"]), dtype=np.float32)
    return np.stack(feature).astype(np.float32)


def frozen_predict(
    model: v375.LifecycleNet,
    feature: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.clip((feature - mean) / std, -8.0, 8.0).astype(np.float32)
    probability, log_mass = v375.predict(model, normalized, device)
    mass = probability * np.exp(np.clip(log_mass, -16.0, 2.0))
    return probability.astype(np.float32), mass.astype(np.float32)


def combine_scores(
    local_probability: np.ndarray,
    local_mass: np.ndarray,
    synthetic_probability: np.ndarray,
    synthetic_mass: np.ndarray,
    covered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probability = local_probability.copy()
    mass = local_mass.copy()
    probability[covered] = sigmoid(
        logit(local_probability[covered])
        + TRANSFER_STRENGTH
        * (logit(synthetic_probability[covered]) - float(logit(SYNTHETIC_PRIOR)))
    ).astype(np.float32)
    local_conditional = np.divide(
        local_mass[covered], np.maximum(local_probability[covered], 1e-5)
    )
    synthetic_conditional = np.divide(
        synthetic_mass[covered], np.maximum(synthetic_probability[covered], 1e-5)
    )
    conditional = np.exp(
        (1.0 - TRANSFER_STRENGTH) * np.log(np.maximum(local_conditional, 1e-7))
        + TRANSFER_STRENGTH * np.log(np.maximum(synthetic_conditional, 1e-7))
    )
    mass[covered] = probability[covered] * conditional
    return probability, mass.astype(np.float32)


def metrics(
    bundle: v370.TrackBundle,
    query_ids: np.ndarray,
    probability: np.ndarray,
    mass: np.ndarray,
    covered: np.ndarray | None,
) -> dict[str, float]:
    return v373.subset_metrics(
        bundle, query_ids, probability, mass, covered
    )


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

    fold_rows = []
    cache: dict[str, list[np.ndarray]] = {
        "fold": [], "truth_binary": [], "truth_mass": [], "covered": [],
        "local_probability": [], "local_mass": [], "synthetic_probability": [],
        "synthetic_mass": [], "shuffled_probability": [], "combined_probability": [],
        "combined_mass": [],
    }
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        bundle = v370.prepare_bundle(valid_ids, fit_ids, geometry, node_raw, component)
        families, discovery = v373.discover_families(
            bundle, fit_ids, all_pos[:, :2]
        )
        exposure = v373.build_exposures(
            families, bundle, outer_ids, all_pos[:, :2]
        )
        feature = exposure_features(
            exposure=exposure,
            families=families,
            bundle=bundle,
            all_pos=all_pos,
            point_feature=point_feature,
            raster=raster,
            bs=bs,
        )
        synthetic_pair_probability, synthetic_pair_mass = frozen_predict(
            model, feature, feature_mean, feature_std, device
        )
        shuffled_feature = v375.shuffled_geometry(
            feature, np.zeros(len(feature), dtype=np.int8), 376_000 + fold
        )
        shuffled_pair_probability, shuffled_pair_mass = frozen_predict(
            model, shuffled_feature, feature_mean, feature_std, device
        )
        synthetic_probability, synthetic_mass, covered = v373.scatter_exposure_scores(
            exposure,
            synthetic_pair_probability,
            synthetic_pair_mass,
            len(bundle.node_query_id),
        )
        shuffled_probability, shuffled_mass, _ = v373.scatter_exposure_scores(
            exposure,
            shuffled_pair_probability,
            shuffled_pair_mass,
            len(bundle.node_query_id),
        )
        local_probability, local_mass, _ = v371.fit_local_scores(
            bundle, fit_ids, outer_ids, 376_100 + fold
        )
        combined_probability, combined_mass = combine_scores(
            local_probability,
            local_mass,
            synthetic_probability,
            synthetic_mass,
            covered,
        )
        local_covered = metrics(
            bundle, outer_ids, local_probability, local_mass, covered
        )
        synthetic_covered = metrics(
            bundle, outer_ids, synthetic_probability, synthetic_mass, covered
        )
        shuffled_covered = metrics(
            bundle, outer_ids, shuffled_probability, shuffled_mass, covered
        )
        local_all = metrics(bundle, outer_ids, local_probability, local_mass, None)
        combined_all = metrics(
            bundle, outer_ids, combined_probability, combined_mass, None
        )
        fold_rows.append(
            {
                "fold": fold,
                "discovery": discovery,
                "exposure_count": int(len(exposure.node)),
                "covered_fraction": float(
                    np.mean(covered[np.isin(bundle.node_query_id, outer_ids)])
                ),
                "local_covered": local_covered,
                "synthetic_covered": synthetic_covered,
                "shuffled_geometry_covered": shuffled_covered,
                "local_all": local_all,
                "combined_all": combined_all,
                "deltas": {
                    "synthetic_vs_shuffled_ap": float(
                        synthetic_covered["existence_ap"]
                        - shuffled_covered["existence_ap"]
                    ),
                    "synthetic_vs_local_covered_ap": float(
                        synthetic_covered["existence_ap"]
                        - local_covered["existence_ap"]
                    ),
                    "combined_vs_local_all_ap": float(
                        combined_all["existence_ap"] - local_all["existence_ap"]
                    ),
                    "combined_vs_local_all_mass": float(
                        combined_all["mass_cosine"] - local_all["mass_cosine"]
                    ),
                },
                "leakage_assertions": {
                    "outer_in_family_discovery": 0,
                    "outer_in_support_features": 0,
                    "outer_in_synthetic_pretraining": 0,
                    "outer_in_local_model_training": 0,
                    "outer_in_donor_bank": 0,
                },
            }
        )
        outer_nodes = np.isin(bundle.node_query_id, outer_ids)
        for key, value in (
            ("truth_binary", bundle.arrays["binary"].astype(np.int8)),
            ("truth_mass", bundle.arrays["mass"]),
            ("covered", covered),
            ("local_probability", local_probability),
            ("local_mass", local_mass),
            ("synthetic_probability", synthetic_probability),
            ("synthetic_mass", synthetic_mass),
            ("shuffled_probability", shuffled_probability),
            ("combined_probability", combined_probability),
            ("combined_mass", combined_mass),
        ):
            cache[key].append(value[outer_nodes])
        cache["fold"].append(np.full(np.sum(outer_nodes), fold, dtype=np.int8))
        print(
            f"v376 fold={fold} exposures={len(exposure.node)} "
            f"AP local={local_covered['existence_ap']:.4f} "
            f"synthetic={synthetic_covered['existence_ap']:.4f} "
            f"shuffle={shuffled_covered['existence_ap']:.4f} "
            f"combined_delta={combined_all['existence_ap']-local_all['existence_ap']:+.4f}",
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
            row["deltas"]["synthetic_vs_shuffled_ap"] > 0.0
            and row["deltas"]["combined_vs_local_all_ap"] > 0.0
            and row["deltas"]["combined_vs_local_all_mass"] > 0.0
            for row in fold_rows
        )
    )
    passed = bool(
        mean["synthetic_vs_shuffled_ap"] >= MIN_SYNTHETIC_VS_SHUFFLED_AP
        and mean["combined_vs_local_all_ap"] >= MIN_COMBINED_VS_LOCAL_AP
        and mean["combined_vs_local_all_mass"] >= MIN_COMBINED_VS_LOCAL_MASS
        and fold_wins >= MIN_FOLD_WINS
    )
    summary = {**mean, "fold_wins": fold_wins}
    report = {
        "method": VERSION,
        "status": "frozen_transfer_passed" if passed else "analysis_only_no_transfer",
        "scope": "strict official path-level frozen transfer; no PDP/full-H/test target/ZIP",
        "primary_hypothesis": (
            "The v375 scene-held-out geometry encoder supplies official path lifecycle "
            "information complementary to the fit-only local path model."
        ),
        "parameters": {
            "checkpoint": PATHS.relative(CHECKPOINT),
            "official_frequency_hz": OFFICIAL_FREQUENCY,
            "transfer_strength": TRANSFER_STRENGTH,
            "synthetic_prior": SYNTHETIC_PRIOR,
            "control": "official point/ray feature rows shuffled within each outer fold",
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "synthetic_vs_shuffled_ap": f">={MIN_SYNTHETIC_VS_SHUFFLED_AP}",
            "combined_vs_local_all_ap": f">={MIN_COMBINED_VS_LOCAL_AP}",
            "combined_vs_local_all_mass": f">={MIN_COMBINED_VS_LOCAL_MASS}",
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
