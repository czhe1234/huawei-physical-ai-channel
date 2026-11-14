#!/usr/bin/env python3
"""Render global PDP from independent patch-specific path energy fields."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v315_full_sequence_confirmatory as v315
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 348
VERSION = "v348_patchwise_path_energy_field"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v348_patchwise_path_energy_oof.npz"

ALPHA_GRID = np.linspace(0.0, 1.25, 11, dtype=np.float32)
FIELD_CONFIGS = (
    (8, 1.0, 0.0),
    (8, 2.0, 0.0),
    (16, 1.0, 0.0),
    (16, 2.0, 0.0),
    (32, 1.0, 0.0),
    (32, 2.0, 0.0),
    (8, 2.0, 0.5),
    (16, 2.0, 0.5),
    (32, 2.0, 0.5),
)
DISTANCE_FLOOR = 0.25
MIN_OFFICIAL_DELTA = 0.01


@dataclass
class PatchBank:
    points: np.ndarray
    candidates: np.ndarray
    tree: cKDTree


def patch_banks(
    candidate: dict[str, np.ndarray], fit_ids: np.ndarray, xy: np.ndarray
) -> dict[int, PatchBank]:
    grouped: dict[int, list[tuple[int, int]]] = {}
    valid = candidate["score"] > 0.0
    for point_id in fit_ids:
        for candidate_id in np.flatnonzero(valid[point_id]):
            patch_id = int(candidate["patch_id"][point_id, candidate_id])
            grouped.setdefault(patch_id, []).append((int(point_id), int(candidate_id)))
    output = {}
    for patch_id, values in grouped.items():
        points = np.asarray([value[0] for value in values], dtype=np.int64)
        candidates = np.asarray([value[1] for value in values], dtype=np.int16)
        output[patch_id] = PatchBank(points, candidates, cKDTree(xy[points]))
    return output


def weighted_mean(
    distance: np.ndarray, value: np.ndarray, power: float
) -> float:
    weight = 1.0 / np.maximum(np.asarray(distance), DISTANCE_FLOOR) ** power
    return float(np.sum(weight * value) / np.maximum(np.sum(weight), 1e-30))


def path_field(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    xy: np.ndarray,
    positions: np.ndarray,
    candidate: dict[str, np.ndarray],
    path_mass: np.ndarray,
    anchor: np.ndarray,
    config: tuple[int, float, float],
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, dict[str, object]]:
    k, distance_power, score_power = config
    banks = patch_banks(candidate, fit_ids, xy)
    direct_tree = cKDTree(xy[fit_ids])
    output = np.empty((len(query_ids), v297.N_DELAY), dtype=np.float32)
    path_mass_sum = np.zeros(len(query_ids), dtype=np.float32)
    reflection_count = np.zeros(len(query_ids), dtype=np.float32)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    valid = candidate["score"] > 0.0

    for row, point_id in enumerate(query_ids):
        distance, local = direct_tree.query(xy[point_id], k=min(k, len(fit_ids)))
        distance = np.atleast_1d(distance)
        local = np.atleast_1d(local)
        direct_mass = weighted_mean(
            distance, path_mass[fit_ids[local], 0].astype(np.float64), distance_power
        )
        delays = [
            intercept + slope * float(np.linalg.norm(positions[point_id] - bs))
        ]
        masses = [max(direct_mass, 0.0)]
        for candidate_id in np.flatnonzero(valid[point_id]):
            patch_id = int(candidate["patch_id"][point_id, candidate_id])
            bank = banks.get(patch_id)
            if bank is None:
                continue
            distance, local = bank.tree.query(
                xy[point_id], k=min(k, len(bank.points))
            )
            distance = np.atleast_1d(distance)
            local = np.atleast_1d(local)
            fit_point = bank.points[local]
            fit_candidate = bank.candidates[local]
            value = path_mass[fit_point, fit_candidate.astype(np.int64) + 1].astype(
                np.float64
            )
            mass = weighted_mean(distance, value, distance_power)
            if score_power > 0.0:
                fit_score = candidate["score"][fit_point, fit_candidate]
                score_reference = weighted_mean(distance, fit_score, distance_power)
                ratio = float(candidate["score"][point_id, candidate_id]) / max(
                    score_reference, 1e-4
                )
                mass *= float(np.clip(ratio, 0.25, 4.0) ** score_power)
            if mass <= 1e-7:
                continue
            delays.append(
                intercept
                + slope * float(candidate["path_length"][point_id, candidate_id])
            )
            masses.append(mass)
        masses_array = np.asarray(masses, dtype=np.float32)
        total = float(np.sum(masses_array))
        if total > 0.95:
            masses_array *= np.float32(0.95 / total)
            total = 0.95
        atoms = v297.dirichlet_atoms(np.asarray(delays, dtype=np.float32)).T
        rendered = np.sum(masses_array[:, None] * atoms, axis=0)
        output[row] = v297.normalize_rows(
            ((1.0 - total) * anchor[row] + rendered)[None, :]
        )[0]
        path_mass_sum[row] = total
        reflection_count[row] = max(len(masses) - 1, 0)
    return output, {
        "path_mass": summarize(path_mass_sum),
        "reflection_count": summarize(reflection_count),
        "active_fraction": float(np.mean(reflection_count > 0.0)),
    }


def blended(anchor: np.ndarray, field: np.ndarray, alpha: float) -> np.ndarray:
    return v297.normalize_rows(anchor + float(alpha) * (field - anchor))


def select_inner(
    anchor: np.ndarray,
    truth: np.ndarray,
    fields: dict[str, np.ndarray],
) -> tuple[str, float, list[dict[str, float]]]:
    rows = []
    for name, value in fields.items():
        for alpha in ALPHA_GRID:
            score = float(np.mean(v297.row_cos(blended(anchor, value, alpha), truth)))
            rows.append({"config": name, "alpha": float(alpha), "pdp": score})
    best = max(rows, key=lambda item: item["pdp"])
    return str(best["config"]), float(best["alpha"]), rows


def main() -> None:
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    xy = train_pos[:, :2]
    target = np.load(v297.TARGET_CACHE)
    truth_d = v297.normalize_rows(target["d"].astype(np.float32))
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        target["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, xy)
    candidate_data = np.load(v306.TRAIN_CANDIDATES)
    candidate = {key: candidate_data[key] for key in candidate_data.files}
    labels, quality, label_report = v306.build_labels()
    path_mass = labels * quality[:, None]
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    slope = float(calibration["slope_bins_per_meter"])
    intercept = float(calibration["intercept_bins"])
    config_map = {
        f"k{k}_p{power:g}_s{score:g}": (k, power, score)
        for k, power, score in FIELD_CONFIGS
    }

    rows = []
    ids_parts = []
    prediction_parts = []
    anchor_parts = []
    truth_parts = []
    for fold, outer_ids in enumerate(folds):
        inner_ids = folds[(fold + 1) % len(folds)]
        inner_fit = np.setdiff1d(valid_ids, np.concatenate([outer_ids, inner_ids]))
        inner_anchor = v306.knn_anchor(inner_ids, inner_fit, train_pos, truth_d)
        inner_fields = {}
        for name, config in config_map.items():
            inner_fields[name], _ = path_field(
                inner_ids,
                inner_fit,
                xy,
                train_pos,
                candidate,
                path_mass,
                inner_anchor,
                config,
                slope,
                intercept,
            )
        selected_name, alpha, inner_scan = select_inner(
            inner_anchor, truth_d[inner_ids], inner_fields
        )

        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        outer_anchor = v306.knn_anchor(outer_ids, outer_fit, train_pos, truth_d)
        outer_field, diagnostics = path_field(
            outer_ids,
            outer_fit,
            xy,
            train_pos,
            candidate,
            path_mass,
            outer_anchor,
            config_map[selected_name],
            slope,
            intercept,
        )
        prediction = blended(outer_anchor, outer_field, alpha)
        base_score = v297.row_cos(outer_anchor, truth_d[outer_ids])
        model_score = v297.row_cos(prediction, truth_d[outer_ids])
        delta = model_score - base_score
        regimes = regime_of(
            np.load(PATHS.shared_features / "height_geometry_cache.npz")["train_blocked"][
                outer_ids
            ]
        ).astype(str)
        rows.append(
            {
                "fold": fold,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "selected_config": selected_name,
                "selected_alpha": alpha,
                "inner_best_pdp": float(max(item["pdp"] for item in inner_scan)),
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
            f"v348 fold={fold} config={selected_name} alpha={alpha:.3f} "
            f"delta={float(np.mean(delta)):+.6f}",
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
        "round": 1,
        "primary_hypothesis": (
            "Independent patch-specific XY energy fields avoid the direct-path softmax "
            "competition of v306 and can recover test-specific global PDP birth/death."
        ),
        "compliance": {
            "uses": [
                "official train H continuous PDP/path labels",
                "official train coordinates",
                "official Round1_Map point-cloud wall candidates",
            ],
            "does_not_use": [
                "v207/v212/synthetic test truth",
                "restored Etoile scene or material labels",
                "hidden or online labels",
            ],
        },
        "parameters": {
            "field_configs": [list(value) for value in FIELD_CONFIGS],
            "alpha_grid": ALPHA_GRID.tolist(),
            "distance_floor": DISTANCE_FLOOR,
            "nested_selection": "next disjoint fold; outer query never used",
        },
        "label": label_report,
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

