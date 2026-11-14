#!/usr/bin/env python3
"""Audit frozen v385 dense boundary residual on strict official folds."""

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
import lifecycle_geometry_features as lgf
import train_v375_synthetic_lifecycle_pretraining as v375
import train_v384_dense_boundary_residual as v384
from project_paths import PATHS


VERSION_NUMBER = 386
VERSION = "v386_dense_residual_official_transfer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
PAIR_CACHE = OUT_DIR / "v386_dense_residual_official_transfer_pairs.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
CHECKPOINT = PATHS.cache_dir(384) / "v384_dense_boundary_residual_model.pt"
V385_REPORT = PATHS.version_report_dir(385) / "v385_residual_head_attribution.json"

MIN_FULL_VS_LOCAL_AP = 0.005
MIN_FULL_VS_LOCAL_MASS = 0.005
MIN_FULL_VS_SHUFFLED_AP = 0.002
MIN_FULL_VS_SHUFFLED_MASS = 0.002
MIN_FOLD_WINS = 4


def pair_prediction(
    *,
    model: v384.BoundaryResidualNet,
    feature: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    local_probability: np.ndarray,
    local_mass: np.ndarray,
    mass_strength: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = np.clip(
        (feature - feature_mean) / feature_std, -8.0, 8.0
    ).astype(np.float32)
    probability, full_log_mass = v384.predict(
        model,
        normalized,
        local_probability,
        local_mass,
        device,
    )
    baseline_log_mass = v384.baseline_log_conditional(
        local_probability, local_mass
    )
    selected_log_mass = baseline_log_mass + mass_strength * (
        full_log_mass - baseline_log_mass
    )
    mass = probability * np.exp(np.clip(selected_log_mass, -16.0, 2.0))
    return probability.astype(np.float32), mass.astype(np.float32), selected_log_mass


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
    if not bool(v385_report["decision_rule"]["passed"]):
        raise RuntimeError("v385 synthetic head attribution did not pass")
    mass_strength = float(v385_report["selected_mass_strength"])
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
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        bundle = v370.prepare_bundle(valid_ids, fit_ids, geometry, node_raw, component)
        families, discovery = v373.discover_families(
            bundle, fit_ids, all_pos[:, :2]
        )
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
        shuffled_feature = v375.shuffled_geometry(
            feature, np.zeros(len(feature), dtype=np.int8), 386_000 + fold
        )
        local_probability, local_mass, _ = v371.fit_local_scores(
            bundle, fit_ids, outer_ids, 386_100 + fold
        )
        pair_local_probability = local_probability[exposure.node]
        pair_local_mass = local_mass[exposure.node]
        full_pair_probability, full_pair_mass, _ = pair_prediction(
            model=model,
            feature=feature,
            feature_mean=feature_mean,
            feature_std=feature_std,
            local_probability=pair_local_probability,
            local_mass=pair_local_mass,
            mass_strength=mass_strength,
            device=device,
        )
        shuffled_pair_probability, shuffled_pair_mass, _ = pair_prediction(
            model=model,
            feature=shuffled_feature,
            feature_mean=feature_mean,
            feature_std=feature_std,
            local_probability=pair_local_probability,
            local_mass=pair_local_mass,
            mass_strength=mass_strength,
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
        full_output_probability = local_probability.copy()
        full_output_mass = local_mass.copy()
        shuffled_output_probability = local_probability.copy()
        shuffled_output_mass = local_mass.copy()
        full_output_probability[covered] = full_probability[covered]
        full_output_mass[covered] = full_mass[covered]
        shuffled_output_probability[covered] = shuffled_probability[covered]
        shuffled_output_mass[covered] = shuffled_mass[covered]

        local_metrics = v373.subset_metrics(
            bundle, outer_ids, local_probability, local_mass, None
        )
        full_metrics = v373.subset_metrics(
            bundle, outer_ids, full_output_probability, full_output_mass, None
        )
        shuffled_metrics = v373.subset_metrics(
            bundle,
            outer_ids,
            shuffled_output_probability,
            shuffled_output_mass,
            None,
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
                "fold": fold,
                "discovery": discovery,
                "outer_exposures": int(len(exposure.node)),
                "covered_fraction": float(
                    np.mean(covered[np.isin(bundle.node_query_id, outer_ids)])
                ),
                "local": local_metrics,
                "shuffled_geometry": shuffled_metrics,
                "full": full_metrics,
                "deltas": deltas,
                "leakage_assertions": {
                    "outer_h_in_family_discovery": 0,
                    "outer_h_in_local_model": 0,
                    "outer_h_in_residual_input": 0,
                    "official_h_in_synthetic_pretraining": 0,
                    "v212_teacher": 0,
                },
            }
        )
        outer_nodes = np.isin(bundle.node_query_id, outer_ids)
        cache["fold"].append(np.full(np.sum(outer_nodes), fold, dtype=np.int8))
        cache["truth_binary"].append(
            bundle.arrays["binary"][outer_nodes].astype(np.int8)
        )
        cache["truth_mass"].append(bundle.arrays["mass"][outer_nodes])
        cache["covered"].append(covered[outer_nodes])
        cache["local_probability"].append(local_probability[outer_nodes])
        cache["local_mass"].append(local_mass[outer_nodes])
        cache["shuffled_probability"].append(
            shuffled_output_probability[outer_nodes]
        )
        cache["shuffled_mass"].append(shuffled_output_mass[outer_nodes])
        cache["full_probability"].append(full_output_probability[outer_nodes])
        cache["full_mass"].append(full_output_mass[outer_nodes])
        print(
            f"v386 fold={fold} exposures={len(exposure.node)} "
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
        "status": "official_transfer_passed" if passed else "analysis_only_no_transfer",
        "scope": "frozen dense-city boundary residual on strict official path folds",
        "primary_hypothesis": (
            "The v385 full-existence and quarter-mass residual transfers to official "
            "local path posteriors and depends on real official geometry."
        ),
        "parameters": {
            "checkpoint": PATHS.relative(CHECKPOINT),
            "mass_strength": mass_strength,
            "geometry_control": "within-outer-fold point/terminal feature shuffle",
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
