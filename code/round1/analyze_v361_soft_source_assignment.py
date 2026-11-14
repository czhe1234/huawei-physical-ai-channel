#!/usr/bin/env python3
"""Strict mass-conserving soft virtual-source assignment experiment."""

from __future__ import annotations

import json

import numpy as np

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v311_effective_source_existence_boundary as v311
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v354_nested_virtual_source_transport as v354
import analyze_v359_fingerprint_subfamily_identifiability as v359
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 361
VERSION = "v361_soft_source_assignment"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v361_soft_source_assignment_oof.npz"

MIN_MEAN_DELTA = 0.003
MIN_FOLD_WINS = 4


def soft_targets(
    observation: v311.Observations,
    query_ids: np.ndarray,
    centers: np.ndarray,
    train_pos: np.ndarray,
    hard_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Distribute fixed hard-assignment mass over all physically plausible sources."""
    target = np.zeros_like(hard_target, dtype=np.float32)
    row_lookup = {int(point_id): row for row, point_id in enumerate(query_ids)}
    selected = np.isin(observation.point_id, query_ids)
    by_point: dict[int, list[int]] = {}
    for row in np.flatnonzero(selected):
        by_point.setdefault(int(observation.point_id[row]), []).append(int(row))

    source_first_length = np.linalg.norm(centers - v304.BS[:2], axis=1)
    source_h = v304.expected_h_bin(
        np.arctan2(centers[:, 1] - v304.BS[1], centers[:, 0] - v304.BS[0])
    )
    candidate_counts: list[int] = []
    component_entropies: list[float] = []
    ambiguous_component_count = 0
    plausible_component_count = 0
    raw_total = 0.0

    for point_id, rows in by_point.items():
        point_row = row_lookup[point_id]
        hard_total = float(np.sum(hard_target[point_row], dtype=np.float64))
        if hard_total <= 0.0:
            continue
        rows_array = np.asarray(rows, dtype=np.int64)
        predicted_length = source_first_length + np.linalg.norm(
            centers - train_pos[point_id, :2], axis=1
        )
        delay_error = np.abs(
            observation.length[rows_array, None] - predicted_length[None, :]
        ) / float(v304.METERS_PER_BIN)
        h_error = v305.circular_distance(
            observation.h[rows_array, None], source_h[None, :], 16.0
        )
        cost = (delay_error / v311.MIN_LENGTH_ERROR) ** 2 + (
            h_error / v311.MIN_H_ERROR
        ) ** 2
        plausible = (
            (delay_error <= v311.MIN_LENGTH_ERROR)
            & (h_error <= v311.MIN_H_ERROR)
        )
        point_soft = np.zeros(len(centers), dtype=np.float64)
        for component_row, observation_row in enumerate(rows_array):
            source_ids = np.flatnonzero(plausible[component_row])
            if not len(source_ids):
                continue
            likelihood = np.exp(-0.5 * cost[component_row, source_ids])
            best_confidence = float(np.max(likelihood))
            probability = likelihood / np.maximum(np.sum(likelihood), 1e-30)
            value = (
                float(observation.mass[observation_row])
                * best_confidence
                * probability
            )
            point_soft[source_ids] += value
            candidate_counts.append(int(len(source_ids)))
            plausible_component_count += 1
            ambiguous_component_count += int(len(source_ids) > 1)
            component_entropies.append(
                float(-np.sum(probability * np.log(np.maximum(probability, 1e-30))))
            )
        point_total = float(np.sum(point_soft))
        raw_total += point_total
        if point_total <= 0.0:
            raise AssertionError("hard source mass has no plausible soft assignment")
        target[point_row] = (point_soft * (hard_total / point_total)).astype(np.float32)

    hard_total_by_point = np.sum(hard_target, axis=1)
    soft_total_by_point = np.sum(target, axis=1)
    max_error = float(np.max(np.abs(hard_total_by_point - soft_total_by_point)))
    if not np.allclose(hard_total_by_point, soft_total_by_point, atol=1e-6):
        raise AssertionError("soft source assignment changed per-point source mass")
    positive = hard_total_by_point > 0.0
    hard_support = np.sum(hard_target > 0.0, axis=1)
    soft_support = np.sum(target > 0.0, axis=1)
    return target, {
        "plausible_component_count": plausible_component_count,
        "ambiguous_component_count": ambiguous_component_count,
        "ambiguous_component_fraction": float(
            ambiguous_component_count / max(plausible_component_count, 1)
        ),
        "candidate_count": summarize(np.asarray(candidate_counts, dtype=np.float32)),
        "component_assignment_entropy": summarize(
            np.asarray(component_entropies, dtype=np.float32)
        ),
        "hard_source_support": summarize(hard_support[positive].astype(np.float32)),
        "soft_source_support": summarize(soft_support[positive].astype(np.float32)),
        "raw_soft_total_before_mass_conservation": raw_total,
        "mass_conservation_max_error": max_error,
    }


def delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: float(left[key] - right[key]) for key in ("sh", "d", "weighted")}


def main() -> None:
    for path in (
        v354.TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v300.FINGERPRINT_CACHE,
        v354.TRAIN_CANDIDATES,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    target = np.load(v354.TARGET_CACHE)
    truth_sh = v354.normalize(target["sh"].astype(np.float32))
    truth_d = v354.normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    train_xy = train_pos[:, :2]
    observation, observation_report = v354.build_physical_observations()
    folds = v315.disjoint_folds(valid_ids, train_xy)

    fold_rows: list[dict[str, object]] = []
    oof_payload: dict[str, np.ndarray] = {}
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        centers, source_report = v311.discover_sources(observation, fit_ids)
        hard_fit = v359.assign_observations(observation, fit_ids, centers, train_pos)
        hard_outer = v359.assign_observations(observation, outer_ids, centers, train_pos)
        soft_fit, soft_fit_report = soft_targets(
            observation, fit_ids, centers, train_pos, hard_fit.target
        )
        soft_outer, soft_outer_report = soft_targets(
            observation, outer_ids, centers, train_pos, hard_outer.target
        )
        neighbor = v354.neighbors(outer_ids, fit_ids, train_xy, False)
        if np.intersect1d(np.unique(neighbor.ids), outer_ids).size:
            raise AssertionError("outer H entered v361 KNN current")
        anchor = v354.anchor_from_neighbors(truth_sh, neighbor)
        hard_prediction, hard_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            hard_fit.target,
            centers,
            train_xy,
        )
        soft_prediction, soft_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            soft_fit,
            centers,
            train_xy,
        )
        hard_oracle, hard_oracle_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            hard_fit.target,
            centers,
            train_xy,
            query_mass_override=hard_outer.target,
        )
        soft_oracle, soft_oracle_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            soft_fit,
            centers,
            train_xy,
            query_mass_override=soft_outer,
        )
        metric = {
            "anchor": v354.metrics(anchor, truth_sh[outer_ids], truth_d[outer_ids]),
            "hard": v354.metrics(
                hard_prediction, truth_sh[outer_ids], truth_d[outer_ids]
            ),
            "soft": v354.metrics(
                soft_prediction, truth_sh[outer_ids], truth_d[outer_ids]
            ),
            "hard_oracle": v354.metrics(
                hard_oracle, truth_sh[outer_ids], truth_d[outer_ids]
            ),
            "soft_oracle": v354.metrics(
                soft_oracle, truth_sh[outer_ids], truth_d[outer_ids]
            ),
        }
        deployed_delta = delta(metric["soft"], metric["hard"])
        oracle_delta = delta(metric["soft_oracle"], metric["hard_oracle"])
        soft_mass_metric = v311.evaluate_prediction(
            v311.knn_prediction(outer_ids, fit_ids, soft_fit, train_pos, k=v354.K_ANCHOR),
            soft_outer,
        )
        hard_mass_metric = v311.evaluate_prediction(
            v311.knn_prediction(
                outer_ids, fit_ids, hard_fit.target, train_pos, k=v354.K_ANCHOR
            ),
            hard_outer.target,
        )
        passed_fold = bool(
            deployed_delta["weighted"] >= MIN_MEAN_DELTA
            and oracle_delta["weighted"] > 0.0
        )
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "source_discovery": source_report,
                "hard_fit_assignment": hard_fit.report,
                "hard_outer_assignment": hard_outer.report,
                "soft_fit_assignment": soft_fit_report,
                "soft_outer_assignment": soft_outer_report,
                "metrics": metric,
                "deployed_delta_soft_vs_hard": deployed_delta,
                "oracle_delta_soft_vs_hard": oracle_delta,
                "source_mass_predictability": {
                    "hard": hard_mass_metric,
                    "soft": soft_mass_metric,
                },
                "passed_fold": passed_fold,
                "diagnostics": {
                    "hard": {
                        key: summarize(value) for key, value in hard_diagnostics.items()
                    },
                    "soft": {
                        key: summarize(value) for key, value in soft_diagnostics.items()
                    },
                    "hard_oracle": {
                        key: summarize(value)
                        for key, value in hard_oracle_diagnostics.items()
                    },
                    "soft_oracle": {
                        key: summarize(value)
                        for key, value in soft_oracle_diagnostics.items()
                    },
                },
                "leakage_assertions": {
                    "outer_in_source_discovery": 0,
                    "outer_in_knn_bank": 0,
                    "outer_soft_target_used_for_scoring_oracle_only": 1,
                    "fit_mass_conservation_max_error": soft_fit_report[
                        "mass_conservation_max_error"
                    ],
                    "outer_mass_conservation_max_error": soft_outer_report[
                        "mass_conservation_max_error"
                    ],
                },
            }
        )
        oof_payload[f"fold_{fold}_ids"] = outer_ids
        oof_payload[f"fold_{fold}_anchor"] = anchor
        oof_payload[f"fold_{fold}_hard"] = hard_prediction
        oof_payload[f"fold_{fold}_soft"] = soft_prediction
        print(
            f"v361 fold={fold} ambiguous={soft_fit_report['ambiguous_component_fraction']:.3f} "
            f"delta={deployed_delta['weighted']:+.6f} "
            f"oracle={oracle_delta['weighted']:+.6f} pass={passed_fold}",
            flush=True,
        )

    mean_delta = {
        key: float(
            np.mean([row["deployed_delta_soft_vs_hard"][key] for row in fold_rows])
        )
        for key in ("sh", "d", "weighted")
    }
    mean_oracle_delta = {
        key: float(np.mean([row["oracle_delta_soft_vs_hard"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    fold_wins = int(sum(row["passed_fold"] for row in fold_rows))
    passed = bool(
        mean_delta["weighted"] >= MIN_MEAN_DELTA
        and mean_oracle_delta["weighted"] > 0.0
        and fold_wins >= MIN_FOLD_WINS
    )
    np.savez_compressed(OOF_OUT, **oof_payload)
    report = {
        "method": VERSION,
        "status": "strict_soft_assignment_passed" if passed else "analysis_only_no_candidate",
        "round": 3,
        "primary_hypothesis": (
            "Hard Hungarian source identity discards physically plausible alternative matches "
            "and creates artificial source birth/death; a per-point-mass-conserving likelihood "
            "posterior should make lifecycle targets smoother without changing energy."
        ),
        "compliance": {
            "uses": [
                "official outer-fit H continuous components",
                "official train coordinates",
                "official Round1_Map point-cloud-derived source observations",
            ],
            "does_not_use": [
                "outer H in source discovery, KNN current, or deployed target",
                "v207/v212/synthetic truth or teacher",
                "restored RT channel/material coefficients",
                "local or online labels",
            ],
        },
        "parameters": {
            "likelihood": "exp(-0.5*((delay/0.5)^2+(H/1.0)^2))",
            "plausibility_window": "same delay<=0.5 and H<=1.0 bins as v311",
            "mass_constraint": "soft source mass sum equals hard source mass for every UE",
            "knn": v354.K_ANCHOR,
            "transport_alpha": 1.0,
            "parameter_search": "none",
            "minimum_mean_delta": MIN_MEAN_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
        },
        "observation": observation_report,
        "fold_rows": fold_rows,
        "decision": {
            "mean_delta_soft_vs_hard": mean_delta,
            "mean_oracle_delta_soft_vs_hard": mean_oracle_delta,
            "fold_wins": fold_wins,
            "passed": passed,
            "next_action": (
                "use soft lifecycle targets in the frozen strict deployment chain"
                if passed
                else "retain hard source identity and reject assignment ambiguity as the main bottleneck"
            ),
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
