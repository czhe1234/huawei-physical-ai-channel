#!/usr/bin/env python3
"""Nested holdout audit of shared wall-scatterer energy fields on the UE plane."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import analyze_v254_route_topology_spectral_field as v254
import analyze_v263_full_wall_scatterer_oracle as v263
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v265_scatterer_delay_angle_oracle as v265
import analyze_v273_planar_barrier_kernel as v273
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 276
VERSION = "v276_shared_scatterer_energy_field"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v276_shared_scatterer_oof.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
NNLS_CACHE = PATHS.cache_dir(264) / "v264_train_scatterer_nnls.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
SIGNATURE_CACHE = PATHS.cache_dir(273) / "v273_planar_barrier_features.npz"
V265_REPORT = PATHS.version_report_dir(265) / "v265_scatterer_delay_angle_oracle.json"

OUTER_SEEDS = (276, 1276, 2276, 3276, 4276)
OUTER_COUNT = 320
INNER_COUNT = 240
ALPHA_GRID = np.linspace(0.0, 1.0, 21, dtype=np.float32)
MIN_DELTA = 5e-4


@dataclass(frozen=True)
class Config:
    k: int
    power: float
    signature_penalty: float

    @property
    def name(self) -> str:
        return f"k{self.k}_p{self.power:g}_sig{self.signature_penalty:g}"


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 1e-12)
    out /= np.maximum(
        out.sum(axis=tuple(range(1, out.ndim)), keepdims=True), 1e-30
    )
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def metrics(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def signature_distance(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    difference = np.sum(query[None, :] != bank, axis=1)
    union = np.sum(query[None, :] | bank, axis=1)
    return difference / np.maximum(union, 1)


def occurrence_index(
    fit_ids: np.ndarray,
    patch_id: np.ndarray,
    valid: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    point_parts: dict[int, list[int]] = {}
    path_parts: dict[int, list[int]] = {}
    for point_id in fit_ids:
        for path_index in np.flatnonzero(valid[point_id]):
            pid = int(patch_id[point_id, path_index])
            point_parts.setdefault(pid, []).append(int(point_id))
            path_parts.setdefault(pid, []).append(int(path_index))
    return {
        pid: (
            np.asarray(point_parts[pid], dtype=np.int64),
            np.asarray(path_parts[pid], dtype=np.int64),
        )
        for pid in point_parts
    }


def weighted_rows(
    query_xy: np.ndarray,
    query_signature: np.ndarray,
    bank_ids: np.ndarray,
    train_xy: np.ndarray,
    signatures: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(train_xy[bank_ids] - query_xy[None, :], axis=1)
    signature = signature_distance(query_signature, signatures[bank_ids])
    effective = distance + config.signature_penalty * signature
    take = min(config.k, len(bank_ids))
    local = np.argpartition(effective, take - 1)[:take]
    selected = bank_ids[local]
    weight = 1.0 / np.maximum(effective[local], 0.05) ** config.power
    weight /= np.maximum(weight.sum(), 1e-20)
    return selected, weight.astype(np.float32)


def predict_weights(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    train_xy: np.ndarray,
    signatures: np.ndarray,
    patch_id: np.ndarray,
    path_valid: np.ndarray,
    nnls_weights: np.ndarray,
    config: Config,
) -> np.ndarray:
    occurrences = occurrence_index(fit_ids, patch_id, path_valid)
    path_weight = nnls_weights[:, : v264.PATHS_PER_POINT * len(v264.SHIFTS)].reshape(
        len(nnls_weights), v264.PATHS_PER_POINT, len(v264.SHIFTS)
    )
    tail_weight = nnls_weights[:, -v264.TAIL_COUNT :]
    output = np.zeros((len(query_ids), nnls_weights.shape[1]), dtype=np.float32)
    for row, query_id in enumerate(query_ids):
        for query_path in np.flatnonzero(path_valid[query_id]):
            pid = int(patch_id[query_id, query_path])
            if pid not in occurrences:
                continue
            occurrence_points, occurrence_paths = occurrences[pid]
            distance = np.linalg.norm(
                train_xy[occurrence_points] - train_xy[query_id][None, :], axis=1
            )
            signature = signature_distance(
                signatures[query_id], signatures[occurrence_points]
            )
            effective = distance + config.signature_penalty * signature
            take = min(config.k, len(occurrence_points))
            local = np.argpartition(effective, take - 1)[:take]
            weight = 1.0 / np.maximum(effective[local], 0.05) ** config.power
            weight /= np.maximum(weight.sum(), 1e-20)
            value = np.sum(
                path_weight[
                    occurrence_points[local], occurrence_paths[local]
                ]
                * weight[:, None],
                axis=0,
            )
            start = query_path * len(v264.SHIFTS)
            output[row, start : start + len(v264.SHIFTS)] = value.astype(np.float32)
        selected, weight = weighted_rows(
            train_xy[query_id],
            signatures[query_id],
            fit_ids,
            train_xy,
            signatures,
            config,
        )
        output[row, -v264.TAIL_COUNT :] = np.sum(
            tail_weight[selected] * weight[:, None], axis=0
        )
        total = float(output[row].sum())
        if total <= 1e-20:
            output[row, -1] = 1.0
        else:
            output[row] /= total
    return output


def render_d(weights: np.ndarray, feature: v264.FeatureSet) -> np.ndarray:
    atoms = v264.fixed_numpy_atoms(
        feature.delay, feature.path_type, feature.path_valid
    )
    return normalize(np.einsum("ba,bad->bd", weights, atoms, optimize=True))


def render_sh(
    weights: np.ndarray,
    query_ids: np.ndarray,
    feature: v264.FeatureSet,
    h_center: np.ndarray,
    path_valid: np.ndarray,
    patch_id: np.ndarray,
    patch_offset: np.ndarray,
    best_spec: dict[str, object],
) -> np.ndarray:
    path_weight, effective_delay = v265.collapse_path_weights(
        weights, feature.delay
    )
    calibrated_h = v265.apply_patch_offsets(
        h_center[query_ids], patch_id[query_ids], patch_offset
    )
    return v265.render_sh(
        path_weight,
        effective_delay,
        calibrated_h,
        float(best_spec["s_sign"]),
        path_valid[query_ids],
        float(best_spec["s_sigma"]),
        float(best_spec["h_sigma"]),
    )


def anchors(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    train_xy: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        v265.knn_anchor(
            train_xy[fit_ids], truth_sh[fit_ids], train_xy[query_ids]
        ),
        v264.knn_anchor(
            train_xy[fit_ids], truth_d[fit_ids], train_xy[query_ids]
        ),
    )


def select_blend(
    anchor_sh: np.ndarray,
    anchor_d: np.ndarray,
    field_sh: np.ndarray,
    field_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = []
    for alpha_sh in ALPHA_GRID:
        blended_sh = normalize(anchor_sh + alpha_sh * (field_sh - anchor_sh))
        sh_score = float(np.mean(row_cos(blended_sh, truth_sh)))
        for alpha_d in ALPHA_GRID:
            blended_d = normalize(anchor_d + alpha_d * (field_d - anchor_d))
            d_score = float(np.mean(row_cos(blended_d, truth_d)))
            rows.append(
                {
                    "alpha_sh": float(alpha_sh),
                    "alpha_d": float(alpha_d),
                    "sh": sh_score,
                    "d": d_score,
                    "weighted": 0.52 * sh_score + 0.48 * d_score,
                }
            )
    return max(rows, key=lambda item: item["weighted"]), rows


def configs() -> list[Config]:
    return [
        Config(k, power, signature_penalty)
        for k in (4, 8, 16)
        for power in (1.0, 2.0)
        for signature_penalty in (0.0, 4.0, 12.0)
    ]


def main() -> None:
    for path in (
        TARGET_CACHE,
        TRAIN_CANDIDATES,
        NNLS_CACHE,
        SCENE_CACHE,
        SIGNATURE_CACHE,
        V265_REPORT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    train_xy = train_pos[:, :2]
    targets = np.load(TARGET_CACHE)
    truth_sh = normalize(targets["sh"].astype(np.float32))
    truth_d = normalize(targets["d"].astype(np.float32))
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    nnls_weights = np.load(NNLS_CACHE)["weights"].astype(np.float32)
    candidates = dict(np.load(TRAIN_CANDIDATES))
    patches = np.load(SCENE_CACHE)["wall_patches"].astype(np.float32)
    signatures = np.load(SIGNATURE_CACHE)["train_signature"].astype(bool)
    dummy_global = np.zeros((len(train_pos), 1), dtype=np.float32)
    train_feature = v264.build_features(
        train_pos, candidates, patches, dummy_global
    )
    azimuth, path_valid, patch_id = v265.departure_azimuth(
        train_pos, candidates
    )
    with V265_REPORT.open("r", encoding="utf-8") as handle:
        best_spec = json.load(handle)["selection"]["best"]
    h_center = v265.h_center_for(
        azimuth,
        float(best_spec["orientation"]),
        str(best_spec["h_mode"]),
        float(best_spec["h_sign"]),
        float(best_spec["h_shift"]),
    )
    true_path_weight, true_delay = v265.collapse_path_weights(
        nnls_weights, train_feature.delay
    )

    all_configs = configs()
    outer_rows: list[dict[str, object]] = []
    oof_ids_parts: list[np.ndarray] = []
    oof_sh_parts: list[np.ndarray] = []
    oof_d_parts: list[np.ndarray] = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_xy, test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = v273.matched_subset(
            outer_fit,
            train_xy,
            test_pos[:, :2],
            INNER_COUNT,
            seed + 10_000,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_patch_offset, _ = v265.estimate_patch_offsets(
            truth_sh,
            inner_fit,
            true_path_weight,
            true_delay,
            h_center,
            float(best_spec["s_sign"]),
            path_valid,
            patch_id,
            len(patches),
        )
        inner_anchor_sh, inner_anchor_d = anchors(
            inner_fit, inner_ids, train_xy, truth_sh, truth_d
        )
        inner_results = []
        for config in all_configs:
            weight = predict_weights(
                inner_fit,
                inner_ids,
                train_xy,
                signatures,
                patch_id,
                path_valid,
                nnls_weights,
                config,
            )
            query_feature = train_feature.take(inner_ids)
            field_d = render_d(weight, query_feature)
            field_sh = render_sh(
                weight,
                inner_ids,
                query_feature,
                h_center,
                path_valid,
                patch_id,
                inner_patch_offset,
                best_spec,
            )
            selected_blend, _ = select_blend(
                inner_anchor_sh,
                inner_anchor_d,
                field_sh,
                field_d,
                truth_sh[inner_ids],
                truth_d[inner_ids],
            )
            inner_results.append(
                {
                    "config": config,
                    "blend": selected_blend,
                }
            )
        selected = max(
            inner_results, key=lambda item: item["blend"]["weighted"]
        )
        selected_config: Config = selected["config"]
        selected_blend = selected["blend"]

        outer_patch_offset, patch_report = v265.estimate_patch_offsets(
            truth_sh,
            outer_fit,
            true_path_weight,
            true_delay,
            h_center,
            float(best_spec["s_sign"]),
            path_valid,
            patch_id,
            len(patches),
        )
        outer_weight = predict_weights(
            outer_fit,
            outer_ids,
            train_xy,
            signatures,
            patch_id,
            path_valid,
            nnls_weights,
            selected_config,
        )
        outer_feature = train_feature.take(outer_ids)
        outer_field_d = render_d(outer_weight, outer_feature)
        outer_field_sh = render_sh(
            outer_weight,
            outer_ids,
            outer_feature,
            h_center,
            path_valid,
            patch_id,
            outer_patch_offset,
            best_spec,
        )
        outer_anchor_sh, outer_anchor_d = anchors(
            outer_fit, outer_ids, train_xy, truth_sh, truth_d
        )
        pred_sh = normalize(
            outer_anchor_sh
            + float(selected_blend["alpha_sh"])
            * (outer_field_sh - outer_anchor_sh)
        )
        pred_d = normalize(
            outer_anchor_d
            + float(selected_blend["alpha_d"])
            * (outer_field_d - outer_anchor_d)
        )
        baseline = metrics(
            outer_anchor_sh,
            outer_anchor_d,
            truth_sh[outer_ids],
            truth_d[outer_ids],
        )
        field_metric = metrics(
            pred_sh, pred_d, truth_sh[outer_ids], truth_d[outer_ids]
        )
        delta = {
            key: field_metric[key] - baseline[key]
            for key in ("sh", "d", "weighted")
        }
        outer_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "selected_config": selected_config.name,
                "selected_blend": selected_blend,
                "inner_top_10": [
                    {
                        "config": item["config"].name,
                        "blend": item["blend"],
                    }
                    for item in sorted(
                        inner_results,
                        key=lambda item: item["blend"]["weighted"],
                        reverse=True,
                    )[:10]
                ],
                "baseline": baseline,
                "field": field_metric,
                "delta": delta,
                "patch_calibration": patch_report,
                "predicted_path_mass": summarize(
                    outer_weight[:, : -v264.TAIL_COUNT].sum(axis=1)
                ),
            }
        )
        oof_ids_parts.append(outer_ids)
        oof_sh_parts.append(pred_sh)
        oof_d_parts.append(pred_d)
        print(
            f"v276 outer {fold + 1}/{len(OUTER_SEEDS)} "
            f"config={selected_config.name} a_sh={selected_blend['alpha_sh']:.2f} "
            f"a_d={selected_blend['alpha_d']:.2f} baseline={baseline['weighted']:.6f} "
            f"field={field_metric['weighted']:.6f} delta={delta['weighted']:+.6f}",
            flush=True,
        )

    delta_weighted = np.asarray(
        [row["delta"]["weighted"] for row in outer_rows], dtype=np.float64
    )
    delta_sh = np.asarray(
        [row["delta"]["sh"] for row in outer_rows], dtype=np.float64
    )
    delta_d = np.asarray(
        [row["delta"]["d"] for row in outer_rows], dtype=np.float64
    )
    selected = bool(
        np.mean(delta_weighted) >= MIN_DELTA
        and np.sum(delta_weighted > 0.0) >= 3
        and np.mean(delta_sh) > 0.0
        and np.mean(delta_d) > 0.0
    )
    np.savez_compressed(
        OOF_CACHE,
        ids=np.concatenate(oof_ids_parts),
        sh=np.concatenate(oof_sh_parts),
        d=np.concatenate(oof_d_parts),
    )
    report = {
        "method": VERSION,
        "status": (
            "official_holdout_passed_next_neuralize"
            if selected
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Per-point NNLS labels become predictable when each stable wall patch is treated "
            "as one shared 2D energy field instead of a rank-position token regressed independently."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H NNLS path-energy labels",
                "official train/test coordinates for matched folds",
                "official point-cloud wall patch IDs and analytic reflection geometry",
                "official point-cloud occlusion component signatures",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher outputs",
                "local replay truth",
                "hidden labels",
                "RT MIMO H or path coefficients",
            ],
            "note": "mechanism audit only; non-neural field is not a submission candidate",
        },
        "parameters": {
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "config_count": len(all_configs),
            "alpha_grid": ALPHA_GRID.tolist(),
            "minimum_delta": MIN_DELTA,
        },
        "outer_folds": outer_rows,
        "selection": {
            "mean_delta": {
                "sh": float(np.mean(delta_sh)),
                "d": float(np.mean(delta_d)),
                "weighted": float(np.mean(delta_weighted)),
            },
            "weighted_wins": int(np.sum(delta_weighted > 0.0)),
            "selected": selected,
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
