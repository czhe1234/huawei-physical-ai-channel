#!/usr/bin/env python3
"""Noise-aware multi-view reliability weighting for virtual-source mass."""

from __future__ import annotations

import json

import numpy as np

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v311_effective_source_existence_boundary as v311
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v354_nested_virtual_source_transport as v354
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 356
VERSION = "v356_noise_aware_virtual_source_mass"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v356_noise_aware_virtual_source_oof.npz"

MIN_MEAN_DELTA = 0.01
MIN_FOLD_WINS = 4


def observation_reliability(
    observation: v311.Observations,
) -> tuple[np.ndarray, dict[str, object]]:
    component = np.load(v297.COMPONENT_CACHE)
    fingerprint = np.load(v300.FINGERPRINT_CACHE)
    point_id = observation.point_id
    component_id = observation.component_id
    point_half = np.clip(
        component["half_profile_cosine"][point_id].astype(np.float32), 0.0, 1.0
    )
    h_half = np.clip(
        fingerprint["h_ue_half_cosine"][point_id, component_id].astype(np.float32),
        0.0,
        1.0,
    )
    v_half = np.clip(
        fingerprint["v_ue_half_cosine"][point_id, component_id].astype(np.float32),
        0.0,
        1.0,
    )
    noise_floor = component["diffuse"][point_id].astype(np.float32) / float(
        v297.N_DELAY
    )
    snr = observation.mass / np.maximum(noise_floor, 1e-8)
    snr_reliability = snr / (1.0 + snr)
    multi_view = np.power(
        np.maximum(point_half * h_half * v_half, 0.0), 1.0 / 3.0
    )
    reliability = np.clip(multi_view * snr_reliability, 0.0, 1.0).astype(
        np.float32
    )
    return reliability, {
        "point_half_profile_cosine": summarize(point_half),
        "horizontal_half_cosine": summarize(h_half),
        "vertical_half_cosine": summarize(v_half),
        "noise_floor": summarize(noise_floor),
        "component_snr": summarize(snr),
        "snr_reliability": summarize(snr_reliability),
        "combined_reliability": summarize(reliability),
    }


def predict_knn_mass(
    fit_ids: np.ndarray,
    fit_target: np.ndarray,
    neighbor: v354.NeighborSet,
    point_count: int,
) -> np.ndarray:
    lookup = np.full(point_count, -1, dtype=np.int64)
    lookup[fit_ids] = np.arange(len(fit_ids), dtype=np.int64)
    rows = lookup[neighbor.ids]
    if np.any(rows < 0):
        raise AssertionError("noise-aware KNN donor outside fit bank")
    return np.sum(
        neighbor.weights[:, :, None] * fit_target[rows],
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)


def main() -> None:
    target = np.load(v354.TARGET_CACHE)
    truth_sh = v354.normalize(target["sh"].astype(np.float32))
    truth_d = v354.normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    observation, observation_report = v354.build_physical_observations()
    reliability, reliability_report = observation_reliability(observation)
    reliable_observation = v311.Observations(
        point=observation.point,
        point_id=observation.point_id,
        component_id=observation.component_id,
        mass=(observation.mass * reliability).astype(np.float32),
        length=observation.length,
        h=observation.h,
    )
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])

    fold_rows: list[dict[str, object]] = []
    oof_ids: list[np.ndarray] = []
    oof_anchor: list[np.ndarray] = []
    oof_prediction: list[np.ndarray] = []
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        centers, source_report = v311.discover_sources(observation, fit_ids)
        fit_target, _ = v311.assign_targets(
            observation, fit_ids, centers, train_pos
        )
        reliable_fit_target, _ = v311.assign_targets(
            reliable_observation, fit_ids, centers, train_pos
        )
        outer_target, _ = v311.assign_targets(
            observation, outer_ids, centers, train_pos
        )
        calibration = np.sum(fit_target, axis=0) / np.maximum(
            np.sum(reliable_fit_target, axis=0), 1e-8
        )
        calibration = np.clip(calibration, 0.5, 2.0).astype(np.float32)
        neighbor = v354.neighbors(
            outer_ids, fit_ids, train_pos[:, :2], False
        )
        if np.intersect1d(np.unique(neighbor.ids), outer_ids).size:
            raise AssertionError("outer H entered v356 KNN")
        anchor = v354.anchor_from_neighbors(truth_sh, neighbor)
        raw_mass = predict_knn_mass(
            fit_ids, fit_target, neighbor, len(train_pos)
        )
        reliable_mass = predict_knn_mass(
            fit_ids, reliable_fit_target, neighbor, len(train_pos)
        ) * calibration[None, :]
        raw_prediction, _raw_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit_target,
            centers,
            train_pos[:, :2],
            query_mass_override=raw_mass,
        )
        prediction, diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit_target,
            centers,
            train_pos[:, :2],
            query_mass_override=reliable_mass,
        )
        base_metric = v354.metrics(
            anchor, truth_sh[outer_ids], truth_d[outer_ids]
        )
        raw_metric = v354.metrics(
            raw_prediction, truth_sh[outer_ids], truth_d[outer_ids]
        )
        model_metric = v354.metrics(
            prediction, truth_sh[outer_ids], truth_d[outer_ids]
        )
        delta = {
            key: float(model_metric[key] - base_metric[key])
            for key in ("sh", "d", "weighted")
        }
        delta_vs_raw = {
            key: float(model_metric[key] - raw_metric[key])
            for key in ("sh", "d", "weighted")
        }
        fold_rows.append(
            {
                "fold": fold,
                "source_discovery": source_report,
                "calibration": summarize(calibration),
                "raw_mass": summarize(raw_mass.sum(axis=1)),
                "reliable_mass": summarize(reliable_mass.sum(axis=1)),
                "true_mass": summarize(outer_target.sum(axis=1)),
                "raw_mass_metrics": v311.evaluate_prediction(raw_mass, outer_target),
                "reliable_mass_metrics": v311.evaluate_prediction(
                    reliable_mass, outer_target
                ),
                "baseline": base_metric,
                "raw_transport": raw_metric,
                "prediction": model_metric,
                "delta": delta,
                "delta_vs_raw_transport": delta_vs_raw,
                "transport": {
                    key: summarize(value) for key, value in diagnostics.items()
                },
                "leakage_assertions": {
                    "outer_in_source_discovery": 0,
                    "outer_in_knn_bank": 0,
                },
            }
        )
        oof_ids.append(outer_ids)
        oof_anchor.append(anchor)
        oof_prediction.append(prediction)
        print(
            f"v356 fold={fold} raw={raw_metric['weighted']-base_metric['weighted']:+.6f} "
            f"noise={delta['weighted']:+.6f} "
            f"vs_raw={delta_vs_raw['weighted']:+.6f}",
            flush=True,
        )

    mean_delta = {
        key: float(np.mean([row["delta"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    mean_vs_raw = {
        key: float(
            np.mean([row["delta_vs_raw_transport"][key] for row in fold_rows])
        )
        for key in ("sh", "d", "weighted")
    }
    wins = int(sum(row["delta"]["weighted"] > 0.0 for row in fold_rows))
    passed = bool(mean_delta["weighted"] >= MIN_MEAN_DELTA and wins >= MIN_FOLD_WINS)
    np.savez_compressed(
        OOF_OUT,
        ids=np.concatenate(oof_ids),
        anchor=np.concatenate(oof_anchor),
        prediction=np.concatenate(oof_prediction),
    )
    report = {
        "method": VERSION,
        "status": "strict_oof_breakthrough" if passed else "analysis_only_no_candidate",
        "round": 3,
        "primary_hypothesis": (
            "Complex-Gaussian-noise contamination makes weak component/source labels "
            "heteroscedastic; independent antenna-half agreement and diffuse floor should "
            "downweight unreliable source mass before local transport."
        ),
        "compliance": {
            "uses": [
                "official train H component mass and independent-half consistency",
                "official fit-only virtual source labels and coordinates",
            ],
            "does_not_use": [
                "outer H in source discovery or KNN",
                "synthetic/v212 teacher, restored RT channel, local or online labels",
            ],
        },
        "parameters": {
            "noise_floor": "per-point diffuse mass / 192",
            "multi_view_reliability": "geometric mean of point/H/V half consistency",
            "fit_only_mass_calibration_clip": [0.5, 2.0],
            "transport": "frozen v354",
            "parameter_search": "none",
            "minimum_mean_delta": MIN_MEAN_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
        },
        "observation": observation_report,
        "reliability": reliability_report,
        "fold_rows": fold_rows,
        "decision": {
            "mean_delta": mean_delta,
            "mean_delta_vs_raw_transport": mean_vs_raw,
            "fold_wins": wins,
            "passed": passed,
        },
        "outputs": {
            "oof": PATHS.relative(OOF_OUT),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
