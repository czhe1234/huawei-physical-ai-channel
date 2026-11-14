#!/usr/bin/env python3
"""Audit label-free quantile alignment before frozen lifecycle transfer."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v373_long_family_visibility as v373
import analyze_v376_frozen_lifecycle_transfer as v376
import analyze_v377_nested_embedding_calibration as v377
import analyze_v379_lifecycle_domain_gap as v379
import lifecycle_geometry_features as lgf
import train_v375_synthetic_lifecycle_pretraining as v375
from project_paths import PATHS


VERSION_NUMBER = 380
VERSION = "v380_quantile_aligned_transfer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
PAIR_CACHE = OUT_DIR / "v380_quantile_aligned_transfer_pairs.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
SYNTHETIC_CACHE = PATHS.cache_dir(375) / "v375_synthetic_lifecycle_examples.npz"
CHECKPOINT = PATHS.cache_dir(375) / "v375_synthetic_lifecycle_model.pt"
OFFICIAL_FEATURE_CACHE = PATHS.cache_dir(379) / "v379_official_lifecycle_features.npz"

QUANTILE_COUNT = 257
MIN_TARGET_STD = 1e-4

MIN_ALIGNED_VS_LOCAL_AP = 0.002
MIN_ALIGNED_VS_LOCAL_MASS = 0.002
MIN_ALIGNED_VS_RAW_AP = 0.002
MIN_ALIGNED_VS_RAW_MASS = 0.002
MIN_ALIGNED_VS_SHUFFLED_AP = 0.002
MIN_ALIGNED_VS_SHUFFLED_MASS = 0.002
MIN_FOLD_WINS = 4


@dataclass
class QuantileAligner:
    probability: np.ndarray
    source_quantile: np.ndarray
    target_quantile: np.ndarray
    active: np.ndarray

    def transform(self, feature: np.ndarray) -> np.ndarray:
        output = feature.copy().astype(np.float32)
        for column in np.flatnonzero(self.active):
            target = self.target_quantile[:, column]
            unique_target, unique_index = np.unique(target, return_index=True)
            if len(unique_target) < 3:
                continue
            rank = np.interp(
                feature[:, column],
                unique_target,
                self.probability[unique_index],
            )
            output[:, column] = np.interp(
                rank, self.probability, self.source_quantile[:, column]
            ).astype(np.float32)
        return output


def fit_aligner(
    synthetic_feature: np.ndarray,
    official_feature: np.ndarray,
) -> QuantileAligner:
    probability = np.linspace(0.0, 1.0, QUANTILE_COUNT, dtype=np.float64)
    source_quantile = np.quantile(synthetic_feature, probability, axis=0).astype(
        np.float32
    )
    target_quantile = np.quantile(official_feature, probability, axis=0).astype(
        np.float32
    )
    active = (
        np.std(synthetic_feature, axis=0) >= MIN_TARGET_STD
    ) & (np.std(official_feature, axis=0) >= MIN_TARGET_STD)
    # Frequency is a known physical input, not an arbitrary domain statistic.
    frequency_column = lgf.feature_dimensions()["point"] + 7
    active[frequency_column] = False
    return QuantileAligner(probability, source_quantile, target_quantile, active)


def feature_representations(
    *,
    feature: np.ndarray,
    aligner: QuantileAligner,
    model: v375.LifecycleNet,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    aligned = aligner.transform(feature)
    shuffled = v375.shuffled_geometry(
        feature, np.zeros(len(feature), dtype=np.int8), seed
    )
    aligned_shuffled = aligner.transform(shuffled)
    return {
        "raw_embedding": v377.frozen_embedding(
            model, feature, feature_mean, feature_std, device
        ),
        "aligned_raw": np.clip(
            (aligned - feature_mean) / feature_std, -8.0, 8.0
        ).astype(np.float32),
        "aligned_shuffled_embedding": v377.frozen_embedding(
            model, aligned_shuffled, feature_mean, feature_std, device
        ),
        "aligned_embedding": v377.frozen_embedding(
            model, aligned, feature_mean, feature_std, device
        ),
    }


def collect_rows(
    rows: list[dict[str, object]],
    *,
    aligner: QuantileAligner,
    model: v375.LifecycleNet,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    seed: int,
    variants: tuple[str, ...],
) -> dict[str, object]:
    output: dict[str, list[np.ndarray]] = {
        "identity": [],
        "binary": [],
        "mass": [],
        "local_probability": [],
        "local_mass": [],
    }
    representation: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    for index, row in enumerate(rows):
        exposure = row["exposure"]
        local_probability = row["local_probability"]
        local_mass = row["local_mass"]
        current = feature_representations(
            feature=row["feature"],
            aligner=aligner,
            model=model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
            seed=seed + index,
        )
        output["identity"].append(exposure.identity)
        output["binary"].append(exposure.binary.astype(np.int8))
        output["mass"].append(exposure.mass)
        output["local_probability"].append(local_probability[exposure.node])
        output["local_mass"].append(local_mass[exposure.node])
        for variant in variants:
            representation[variant].append(current[variant])
    merged: dict[str, object] = {
        key: np.concatenate(value) for key, value in output.items()
    }
    merged["representation"] = {
        key: np.concatenate(value) for key, value in representation.items()
    }
    return merged


def main() -> None:
    for path in (
        TARGET_CACHE,
        SYNTHETIC_CACHE,
        CHECKPOINT,
        OFFICIAL_FEATURE_CACHE,
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

    synthetic_cache = np.load(SYNTHETIC_CACHE)
    synthetic_feature = synthetic_cache["feature"][
        np.isin(synthetic_cache["scene"], np.arange(18))
    ].astype(np.float32)
    official_cache = np.load(OFFICIAL_FEATURE_CACHE)
    official_feature = official_cache["feature"].astype(np.float32)
    aligner = fit_aligner(synthetic_feature, official_feature)

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
        "raw_embedding",
        "aligned_raw",
        "aligned_shuffled_embedding",
        "aligned_embedding",
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
                v377.build_fold_examples(
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
                    seed=380_000 + outer_fold * 100 + inner_fold,
                )
            )
        outer = v377.build_fold_examples(
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
            seed=380_500 + outer_fold,
        )
        train = collect_rows(
            nested_rows,
            aligner=aligner,
            model=model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
            seed=380_600 + outer_fold * 10,
            variants=variants,
        )
        query = collect_rows(
            [outer],
            aligner=aligner,
            model=model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
            seed=380_700 + outer_fold,
            variants=variants,
        )

        outer_bundle = outer["bundle"]
        outer_exposure = outer["exposure"]
        local_probability = outer["local_probability"]
        local_mass = outer["local_mass"]
        local_metrics = v373.subset_metrics(
            outer_bundle, outer_ids, local_probability, local_mass, None
        )
        outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        metrics: dict[str, dict[str, float]] = {}
        diagnostics: dict[str, dict[str, object]] = {}
        for variant in variants:
            train_design = v377.design_matrix(
                train["identity"], train["representation"][variant]
            )
            query_design = v377.design_matrix(
                query["identity"], query["representation"][variant]
            )
            pair_probability, pair_mass, current_diagnostics = v377.fit_calibrator(
                train_design=train_design,
                train_local_probability=train["local_probability"],
                train_local_mass=train["local_mass"],
                train_binary=train["binary"],
                train_mass=train["mass"],
                query_design=query_design,
                query_local_probability=query["local_probability"],
                query_local_mass=query["local_mass"],
            )
            output = v377.apply_calibrated_exposures(
                exposure=outer_exposure,
                probability=pair_probability,
                mass=pair_mass,
                local_probability=local_probability,
                local_mass=local_mass,
            )
            outputs[variant] = output
            metrics[variant] = v373.subset_metrics(
                outer_bundle, outer_ids, output[0], output[1], None
            )
            diagnostics[variant] = current_diagnostics

        aligned = metrics["aligned_embedding"]
        raw = metrics["raw_embedding"]
        shuffled = metrics["aligned_shuffled_embedding"]
        deltas = {
            "aligned_vs_local_ap": float(
                aligned["existence_ap"] - local_metrics["existence_ap"]
            ),
            "aligned_vs_local_mass": float(
                aligned["mass_cosine"] - local_metrics["mass_cosine"]
            ),
            "aligned_vs_raw_ap": float(
                aligned["existence_ap"] - raw["existence_ap"]
            ),
            "aligned_vs_raw_mass": float(
                aligned["mass_cosine"] - raw["mass_cosine"]
            ),
            "aligned_vs_shuffled_ap": float(
                aligned["existence_ap"] - shuffled["existence_ap"]
            ),
            "aligned_vs_shuffled_mass": float(
                aligned["mass_cosine"] - shuffled["mass_cosine"]
            ),
        }
        fold_rows.append(
            {
                "fold": outer_fold,
                "nested_training_exposures": int(len(train["binary"])),
                "outer_exposures": int(len(outer_exposure.node)),
                "local": local_metrics,
                "variants": metrics,
                "calibration": diagnostics,
                "deltas": deltas,
                "leakage_assertions": {
                    "h_label_in_quantile_alignment": 0,
                    "outer_in_nested_calibration": 0,
                    "outer_h_in_family_discovery": 0,
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
        cache["local_probability"].append(local_probability[outer_nodes])
        cache["local_mass"].append(local_mass[outer_nodes])
        cache["covered"].append(outputs["aligned_embedding"][2][outer_nodes])
        for variant in variants:
            cache[f"{variant}_probability"].append(outputs[variant][0][outer_nodes])
            cache[f"{variant}_mass"].append(outputs[variant][1][outer_nodes])
        print(
            f"v380 fold={outer_fold} train={len(train['binary'])} "
            f"delta_local={deltas['aligned_vs_local_ap']:+.4f}/"
            f"{deltas['aligned_vs_local_mass']:+.4f} "
            f"delta_raw={deltas['aligned_vs_raw_ap']:+.4f}/"
            f"{deltas['aligned_vs_raw_mass']:+.4f}",
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
            row["deltas"]["aligned_vs_local_ap"] > 0.0
            and row["deltas"]["aligned_vs_local_mass"] > 0.0
            and row["deltas"]["aligned_vs_raw_ap"] > 0.0
            and row["deltas"]["aligned_vs_raw_mass"] > 0.0
            and row["deltas"]["aligned_vs_shuffled_ap"] > 0.0
            and row["deltas"]["aligned_vs_shuffled_mass"] > 0.0
            for row in fold_rows
        )
    )
    passed = bool(
        mean["aligned_vs_local_ap"] >= MIN_ALIGNED_VS_LOCAL_AP
        and mean["aligned_vs_local_mass"] >= MIN_ALIGNED_VS_LOCAL_MASS
        and mean["aligned_vs_raw_ap"] >= MIN_ALIGNED_VS_RAW_AP
        and mean["aligned_vs_raw_mass"] >= MIN_ALIGNED_VS_RAW_MASS
        and mean["aligned_vs_shuffled_ap"] >= MIN_ALIGNED_VS_SHUFFLED_AP
        and mean["aligned_vs_shuffled_mass"] >= MIN_ALIGNED_VS_SHUFFLED_MASS
        and fold_wins >= MIN_FOLD_WINS
    )
    summary = {
        **mean,
        "fold_wins": fold_wins,
        "aligned_columns": int(np.sum(aligner.active)),
    }
    report = {
        "method": VERSION,
        "status": "aligned_transfer_passed" if passed else "analysis_only_no_transfer",
        "scope": "label-free target geometry alignment plus strict official calibration",
        "primary_hypothesis": (
            "The v375 geometry ordering becomes complementary after monotone marginal "
            "alignment removes the v379 high-density covariate shift."
        ),
        "parameters": {
            "quantile_count": QUANTILE_COUNT,
            "minimum_target_std": MIN_TARGET_STD,
            "source_distribution": "v375 training scenes 0-17",
            "target_distribution": "v379 all official outer features; geometry only",
            "frequency_aligned": False,
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "aligned_vs_local_ap": f">={MIN_ALIGNED_VS_LOCAL_AP}",
            "aligned_vs_local_mass": f">={MIN_ALIGNED_VS_LOCAL_MASS}",
            "aligned_vs_raw_ap": f">={MIN_ALIGNED_VS_RAW_AP}",
            "aligned_vs_raw_mass": f">={MIN_ALIGNED_VS_RAW_MASS}",
            "aligned_vs_shuffled_ap": f">={MIN_ALIGNED_VS_SHUFFLED_AP}",
            "aligned_vs_shuffled_mass": f">={MIN_ALIGNED_VS_SHUFFLED_MASS}",
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
