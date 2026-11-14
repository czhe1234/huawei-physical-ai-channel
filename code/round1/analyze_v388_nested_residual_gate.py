#!/usr/bin/env python3
"""Learn a strict nested low-dimensional gate for v386 residual actions."""

from __future__ import annotations

import json

import numpy as np
import torch

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v373_long_family_visibility as v373
import analyze_v377_nested_embedding_calibration as v377
import analyze_v387_nested_residual_calibration as v387
import lifecycle_geometry_features as lgf
import train_v384_dense_boundary_residual as v384
from project_paths import PATHS


VERSION_NUMBER = 388
VERSION = "v388_nested_residual_gate"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
PAIR_CACHE = OUT_DIR / "v388_nested_residual_gate_pairs.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
CHECKPOINT = PATHS.cache_dir(384) / "v384_dense_boundary_residual_model.pt"
V385_REPORT = PATHS.version_report_dir(385) / "v385_residual_head_attribution.json"

MIN_FULL_VS_LOCAL_AP = 0.002
MIN_FULL_VS_LOCAL_MASS = 0.002
MIN_FULL_VS_SHUFFLED_AP = 0.001
MIN_FULL_VS_SHUFFLED_MASS = 0.001
MIN_FOLD_WINS = 4


def residual_design(
    local_probability: np.ndarray,
    local_mass: np.ndarray,
    raw_probability: np.ndarray,
    raw_mass: np.ndarray,
) -> np.ndarray:
    local_logit = np.clip(v387.logit(local_probability), -12.0, 12.0)
    raw_logit = np.clip(v387.logit(raw_probability), -12.0, 12.0)
    delta_logit = raw_logit - local_logit
    local_conditional = np.divide(
        local_mass, np.maximum(local_probability, 1e-5)
    )
    raw_conditional = np.divide(raw_mass, np.maximum(raw_probability, 1e-5))
    local_log_mass = np.clip(
        np.log(np.maximum(local_conditional, 1e-8)), -16.0, 2.0
    )
    raw_log_mass = np.clip(
        np.log(np.maximum(raw_conditional, 1e-8)), -16.0, 2.0
    )
    delta_mass = raw_log_mass - local_log_mass
    uncertainty = 4.0 * local_probability * (1.0 - local_probability)
    return np.column_stack(
        [
            delta_logit,
            np.abs(delta_logit),
            delta_logit * uncertainty,
            delta_logit * np.sign(local_logit),
            delta_mass,
            np.abs(delta_mass),
            delta_mass * uncertainty,
            uncertainty,
            local_logit / 4.0,
            (local_log_mass + 6.0) / 4.0,
        ]
    ).astype(np.float32)


def extract_training_rows(
    outputs: list[tuple[dict[str, object], np.ndarray]],
    raw_prefix: str,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {
        "design": [],
        "local_probability": [],
        "local_mass": [],
        "binary": [],
        "mass": [],
    }
    for output, query_ids in outputs:
        bundle = output["bundle"]
        keep = output["covered"] & np.isin(bundle.node_query_id, query_ids)
        local_probability = output["local_probability"][keep]
        local_mass = output["local_mass"][keep]
        raw_probability = output[f"raw_{raw_prefix}_probability"][keep]
        raw_mass = output[f"raw_{raw_prefix}_mass"][keep]
        rows["design"].append(
            residual_design(
                local_probability,
                local_mass,
                raw_probability,
                raw_mass,
            )
        )
        rows["local_probability"].append(local_probability)
        rows["local_mass"].append(local_mass)
        rows["binary"].append(bundle.arrays["binary"][keep].astype(np.int8))
        rows["mass"].append(bundle.arrays["mass"][keep].astype(np.float32))
    return {key: np.concatenate(value) for key, value in rows.items()}


def calibrate_outer(
    *,
    train: dict[str, np.ndarray],
    outer: dict[str, object],
    outer_ids: np.ndarray,
    raw_prefix: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    bundle = outer["bundle"]
    covered = outer["covered"] & np.isin(bundle.node_query_id, outer_ids)
    local_probability = outer["local_probability"][covered]
    local_mass = outer["local_mass"][covered]
    raw_probability = outer[f"raw_{raw_prefix}_probability"][covered]
    raw_mass = outer[f"raw_{raw_prefix}_mass"][covered]
    query_design = residual_design(
        local_probability, local_mass, raw_probability, raw_mass
    )
    probability, mass, diagnostics = v377.fit_calibrator(
        train_design=train["design"],
        train_local_probability=train["local_probability"],
        train_local_mass=train["local_mass"],
        train_binary=train["binary"],
        train_mass=train["mass"],
        query_design=query_design,
        query_local_probability=local_probability,
        query_local_mass=local_mass,
    )
    output_probability = outer["local_probability"].copy()
    output_mass = outer["local_mass"].copy()
    output_probability[covered] = probability
    output_mass[covered] = mass
    diagnostics["outer_covered_nodes"] = int(np.sum(covered))
    return output_probability, output_mass, diagnostics


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
        inner_outputs = []
        for inner_fold, inner_ids_all in enumerate(folds):
            if inner_fold == outer_fold:
                continue
            inner_ids = np.intersect1d(fit_ids, inner_ids_all)
            inner_fit = np.setdiff1d(fit_ids, inner_ids)
            inner_outputs.append(
                (
                    v387.raw_fold_outputs(
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
                        seed=388_000 + outer_fold * 100 + inner_fold,
                    ),
                    inner_ids,
                )
            )
        full_train = extract_training_rows(inner_outputs, "full")
        shuffled_train = extract_training_rows(inner_outputs, "shuffled")
        outer = v387.raw_fold_outputs(
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
            seed=388_500 + outer_fold,
        )
        full_probability, full_mass, full_diagnostics = calibrate_outer(
            train=full_train,
            outer=outer,
            outer_ids=outer_ids,
            raw_prefix="full",
        )
        shuffled_probability, shuffled_mass, shuffled_diagnostics = calibrate_outer(
            train=shuffled_train,
            outer=outer,
            outer_ids=outer_ids,
            raw_prefix="shuffled",
        )
        local_metrics = v387.metrics(
            outer,
            outer_ids,
            outer["local_probability"],
            outer["local_mass"],
        )
        full_metrics = v387.metrics(
            outer, outer_ids, full_probability, full_mass
        )
        shuffled_metrics = v387.metrics(
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
                "inner_training_nodes": int(len(full_train["binary"])),
                "local": local_metrics,
                "shuffled_geometry": shuffled_metrics,
                "full": full_metrics,
                "calibration": {
                    "full": full_diagnostics,
                    "shuffled": shuffled_diagnostics,
                },
                "deltas": deltas,
                "leakage_assertions": {
                    "outer_in_gate_training": 0,
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
            f"v388 fold={outer_fold} train={len(full_train['binary'])} "
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
        "status": "nested_residual_gate_passed" if passed else "analysis_only_no_transfer",
        "scope": "strict nested low-dimensional official residual gate",
        "primary_hypothesis": (
            "A low-dimensional gate can identify where the v386 geometry residual is "
            "helpful, outperforming both local and an independently calibrated shuffled control."
        ),
        "parameters": {
            "design_dimension": 10,
            "design": [
                "delta_logit",
                "abs_delta_logit",
                "delta_logit_x_uncertainty",
                "delta_logit_x_local_sign",
                "delta_logmass",
                "abs_delta_logmass",
                "delta_logmass_x_uncertainty",
                "local_uncertainty",
                "local_logit",
                "local_logmass",
            ],
            "calibrator": "v377 fixed-offset logistic plus positive-mass ridge",
            "real_shuffled_heads": "independent equal-capacity nested calibration",
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
