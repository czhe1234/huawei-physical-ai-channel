#!/usr/bin/env python3
"""Strict family-specific residual renderer with parent mass held fixed."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v311_effective_source_existence_boundary as v311
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v354_nested_virtual_source_transport as v354
import analyze_v359_fingerprint_subfamily_identifiability as v359
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 360
VERSION = "v360_fingerprint_subfamily_renderer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v360_fingerprint_subfamily_renderer_oof.npz"

MIN_MEAN_DELTA_VS_V354 = 0.003
MIN_FOLD_WINS = 4


def family_offsets(
    assigned: v359.AssignedObservations,
    family_label: np.ndarray,
    family_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    delay = np.zeros(family_count, dtype=np.float32)
    horizontal = np.zeros(family_count, dtype=np.float32)
    count = np.zeros(family_count, dtype=np.int64)
    mass = np.zeros(family_count, dtype=np.float64)
    for family in range(family_count):
        take = family_label == family
        count[family] = int(np.sum(take))
        if not np.any(take):
            continue
        weight = assigned.value[take].astype(np.float64)
        weight /= np.maximum(np.sum(weight), 1e-30)
        delay[family] = float(np.sum(weight * assigned.delay_residual[take]))
        horizontal[family] = float(np.sum(weight * assigned.h_residual[take]))
        mass[family] = float(np.sum(assigned.value[take], dtype=np.float64))
    if np.any(count == 0):
        raise AssertionError("fingerprint family without fit observations")
    delay = np.clip(delay, -v311.MIN_LENGTH_ERROR, v311.MIN_LENGTH_ERROR)
    horizontal = np.clip(horizontal, -v311.MIN_H_ERROR, v311.MIN_H_ERROR)
    return delay, horizontal, {
        "count": summarize(count.astype(np.float32)),
        "mass": summarize(mass.astype(np.float32)),
        "delay_residual_bins": summarize(delay),
        "h_residual_bins": summarize(horizontal),
    }


def nearest_supported_family(
    query_ids: np.ndarray,
    fit: v359.AssignedObservations,
    family_label: np.ndarray,
    family_parent: np.ndarray,
    observation: v311.Observations,
    train_xy: np.ndarray,
    parent_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.full((len(query_ids), parent_count), -1, dtype=np.int64)
    distance = np.full((len(query_ids), parent_count), np.inf, dtype=np.float32)
    for parent in range(parent_count):
        rows = np.flatnonzero(fit.parent == parent)
        if not len(rows):
            raise AssertionError("parent source without fit support")
        point_ids = observation.point_id[fit.rows[rows]]
        value, local = cKDTree(train_xy[point_ids]).query(train_xy[query_ids], k=1)
        nearest_rows = rows[np.asarray(local, dtype=np.int64)]
        selected[:, parent] = family_label[nearest_rows]
        distance[:, parent] = np.asarray(value, dtype=np.float32)
    if np.any(selected < 0) or np.any(family_parent[selected] != np.arange(parent_count)[None, :]):
        raise AssertionError("nearest support selected a family from another parent")
    return selected, distance


def query_family_mass(
    parent_mass: np.ndarray,
    selected_family: np.ndarray,
    family_count: int,
) -> np.ndarray:
    output = np.zeros((len(parent_mass), family_count), dtype=np.float32)
    rows = np.arange(len(parent_mass), dtype=np.int64)[:, None]
    np.add.at(output, (np.broadcast_to(rows, selected_family.shape), selected_family), parent_mass)
    if not np.allclose(output.sum(axis=1), parent_mass.sum(axis=1), atol=1e-6):
        raise AssertionError("query family selector changed parent total mass")
    return output


def render_family_sources(
    mass: np.ndarray,
    parent_delay: np.ndarray,
    parent_h: np.ndarray,
    family_parent: np.ndarray,
    delay_offset: np.ndarray,
    h_offset: np.ndarray,
) -> np.ndarray:
    delay = parent_delay[:, family_parent] + delay_offset[None, :]
    horizontal = parent_h[family_parent] + h_offset
    return v354.render_query_sources(mass, delay, horizontal)


def family_transport(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    neighbor: v354.NeighborSet,
    anchor: np.ndarray,
    fit_parent_target: np.ndarray,
    fit_family_target: np.ndarray,
    selected_family: np.ndarray,
    family_parent: np.ndarray,
    delay_offset: np.ndarray,
    h_offset: np.ndarray,
    centers: np.ndarray,
    train_xy: np.ndarray,
    query_family_override: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    fit_lookup = np.full(len(train_xy), -1, dtype=np.int64)
    fit_lookup[fit_ids] = np.arange(len(fit_ids), dtype=np.int64)
    donor_rows = fit_lookup[neighbor.ids]
    if np.any(donor_rows < 0):
        raise AssertionError("family renderer donor outside outer-fit bank")

    predicted_parent_mass = np.sum(
        neighbor.weights[:, :, None] * fit_parent_target[donor_rows],
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    predicted_family_mass = query_family_mass(
        predicted_parent_mass, selected_family, len(family_parent)
    )
    rendered_mass = (
        predicted_family_mass
        if query_family_override is None
        else np.asarray(query_family_override, dtype=np.float32)
    )
    query_parent_delay, parent_h = v354.source_geometry(train_xy[query_ids], centers)
    query_source = render_family_sources(
        rendered_mass,
        query_parent_delay,
        parent_h,
        family_parent,
        delay_offset,
        h_offset,
    )

    family_h = parent_h[family_parent] + h_offset
    family_h_atom = v354.exact_atoms(family_h, v354.H_SIZE)
    donor_source = np.zeros_like(anchor, dtype=np.float32)
    for start in range(0, len(query_ids), v354.BATCH_SIZE):
        end = min(start + v354.BATCH_SIZE, len(query_ids))
        ids = neighbor.ids[start:end]
        rows = donor_rows[start:end]
        donor_parent_delay, _ = v354.source_geometry(train_xy[ids.reshape(-1)], centers)
        donor_parent_delay = donor_parent_delay.reshape(
            end - start, ids.shape[1], len(centers)
        )
        donor_family_delay = (
            donor_parent_delay[:, :, family_parent] + delay_offset[None, None, :]
        )
        s_atom = v354.exact_atoms(
            np.remainder(-donor_family_delay, v354.S_SIZE), v354.S_SIZE
        )
        donor_source[start:end] = np.einsum(
            "qk,qkf,qkfd,fh->qdh",
            neighbor.weights[start:end],
            fit_family_target[rows],
            s_atom,
            family_h_atom,
            optimize=True,
        ).astype(np.float32)
    unnormalized = anchor + query_source - donor_source
    prediction = v354.normalize(unnormalized)
    return prediction, {
        "predicted_parent_mass": predicted_parent_mass.sum(axis=1).astype(np.float32),
        "predicted_family_mass": predicted_family_mass.sum(axis=1).astype(np.float32),
        "rendered_query_mass": query_source.sum(axis=(1, 2)).astype(np.float32),
        "donor_mass": donor_source.sum(axis=(1, 2)).astype(np.float32),
        "clipped_negative_mass": np.maximum(-unnormalized, 0.0)
        .sum(axis=(1, 2))
        .astype(np.float32),
    }


def metric_delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: float(left[key] - right[key]) for key in ("sh", "d", "weighted")}


def main() -> None:
    for path in (
        v354.TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v300.FINGERPRINT_CACHE,
        v354.TRAIN_CANDIDATES,
        v273.SCENE_CACHE,
        v273.FEATURE_CACHE,
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
    fingerprint_data = np.load(v300.FINGERPRINT_CACHE)
    fingerprint = {key: fingerprint_data[key] for key in fingerprint_data.files}
    folds = v315.disjoint_folds(valid_ids, train_xy)

    fold_rows: list[dict[str, object]] = []
    oof_payload: dict[str, np.ndarray] = {}
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        centers, source_report = v311.discover_sources(observation, fit_ids)
        fit = v359.assign_observations(observation, fit_ids, centers, train_pos)
        outer = v359.assign_observations(observation, outer_ids, centers, train_pos)
        family = v359.define_families(
            fit,
            outer,
            observation,
            centers,
            fingerprint,
            v359.VERSION_NUMBER * 10_000 + fold,
        )
        real_delay, real_h, real_offset_report = family_offsets(
            fit, family.fit_family, len(family.parent)
        )
        shuffle_delay, shuffle_h, shuffle_offset_report = family_offsets(
            fit, family.fit_family_shuffled, len(family.parent)
        )
        real_selected, real_distance = nearest_supported_family(
            outer_ids,
            fit,
            family.fit_family,
            family.parent,
            observation,
            train_xy,
            len(centers),
        )
        shuffle_selected, shuffle_distance = nearest_supported_family(
            outer_ids,
            fit,
            family.fit_family_shuffled,
            family.parent,
            observation,
            train_xy,
            len(centers),
        )
        neighbor = v354.neighbors(outer_ids, fit_ids, train_xy, False)
        if np.intersect1d(np.unique(neighbor.ids), outer_ids).size:
            raise AssertionError("outer H entered v360 KNN current")
        anchor = v354.anchor_from_neighbors(truth_sh, neighbor)
        parent_prediction, parent_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit.target,
            centers,
            train_xy,
        )
        parent_oracle, parent_oracle_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit.target,
            centers,
            train_xy,
            query_mass_override=outer.target,
        )
        real_prediction, real_diagnostics = family_transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit.target,
            family.fit_target,
            real_selected,
            family.parent,
            real_delay,
            real_h,
            centers,
            train_xy,
        )
        shuffle_prediction, shuffle_diagnostics = family_transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit.target,
            family.shuffled_fit_target,
            shuffle_selected,
            family.parent,
            shuffle_delay,
            shuffle_h,
            centers,
            train_xy,
        )
        family_oracle, family_oracle_diagnostics = family_transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit.target,
            family.fit_target,
            real_selected,
            family.parent,
            real_delay,
            real_h,
            centers,
            train_xy,
            query_family_override=family.outer_target,
        )

        metric = {
            "anchor": v354.metrics(anchor, truth_sh[outer_ids], truth_d[outer_ids]),
            "v354_parent": v354.metrics(
                parent_prediction, truth_sh[outer_ids], truth_d[outer_ids]
            ),
            "fingerprint_family": v354.metrics(
                real_prediction, truth_sh[outer_ids], truth_d[outer_ids]
            ),
            "shuffled_family": v354.metrics(
                shuffle_prediction, truth_sh[outer_ids], truth_d[outer_ids]
            ),
            "v354_parent_oracle": v354.metrics(
                parent_oracle, truth_sh[outer_ids], truth_d[outer_ids]
            ),
            "fingerprint_family_oracle": v354.metrics(
                family_oracle, truth_sh[outer_ids], truth_d[outer_ids]
            ),
        }
        delta_vs_v354 = metric_delta(metric["fingerprint_family"], metric["v354_parent"])
        delta_vs_shuffle = metric_delta(
            metric["fingerprint_family"], metric["shuffled_family"]
        )
        oracle_delta = metric_delta(
            metric["fingerprint_family_oracle"], metric["v354_parent_oracle"]
        )
        passed_fold = bool(
            delta_vs_v354["weighted"] >= MIN_MEAN_DELTA_VS_V354
            and delta_vs_shuffle["weighted"] > 0.0
            and oracle_delta["weighted"] > 0.0
        )
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "source_discovery": source_report,
                "family_definition": family.report,
                "real_offsets": real_offset_report,
                "shuffle_offsets": shuffle_offset_report,
                "nearest_support_distance": summarize(real_distance),
                "shuffle_nearest_support_distance": summarize(shuffle_distance),
                "metrics": metric,
                "delta_vs_v354": delta_vs_v354,
                "delta_vs_shuffle": delta_vs_shuffle,
                "oracle_delta_vs_parent_oracle": oracle_delta,
                "passed_fold": passed_fold,
                "diagnostics": {
                    "v354_parent": {
                        key: summarize(value) for key, value in parent_diagnostics.items()
                    },
                    "fingerprint_family": {
                        key: summarize(value) for key, value in real_diagnostics.items()
                    },
                    "shuffled_family": {
                        key: summarize(value) for key, value in shuffle_diagnostics.items()
                    },
                    "v354_parent_oracle": {
                        key: summarize(value)
                        for key, value in parent_oracle_diagnostics.items()
                    },
                    "fingerprint_family_oracle": {
                        key: summarize(value)
                        for key, value in family_oracle_diagnostics.items()
                    },
                },
                "leakage_assertions": {
                    "outer_in_source_discovery": 0,
                    "outer_in_knn_or_family_support": 0,
                    "outer_fingerprint_used_as_prediction_input": 0,
                    "outer_fingerprint_used_for_oracle_scoring_only": 1,
                    "query_parent_mass_changed": 0,
                },
            }
        )
        oof_payload[f"fold_{fold}_ids"] = outer_ids
        oof_payload[f"fold_{fold}_anchor"] = anchor
        oof_payload[f"fold_{fold}_v354_parent"] = parent_prediction
        oof_payload[f"fold_{fold}_fingerprint_family"] = real_prediction
        oof_payload[f"fold_{fold}_shuffled_family"] = shuffle_prediction
        print(
            f"v360 fold={fold} family={len(family.parent)} "
            f"delta={delta_vs_v354['weighted']:+.6f} "
            f"shuffle={delta_vs_shuffle['weighted']:+.6f} "
            f"oracle={oracle_delta['weighted']:+.6f} pass={passed_fold}",
            flush=True,
        )

    mean_delta_vs_v354 = {
        key: float(np.mean([row["delta_vs_v354"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    mean_delta_vs_shuffle = {
        key: float(np.mean([row["delta_vs_shuffle"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    mean_oracle_delta = {
        key: float(
            np.mean([row["oracle_delta_vs_parent_oracle"][key] for row in fold_rows])
        )
        for key in ("sh", "d", "weighted")
    }
    fold_wins = int(sum(row["passed_fold"] for row in fold_rows))
    passed = bool(
        mean_delta_vs_v354["weighted"] >= MIN_MEAN_DELTA_VS_V354
        and mean_delta_vs_shuffle["weighted"] > 0.0
        and mean_oracle_delta["weighted"] > 0.0
        and fold_wins >= MIN_FOLD_WINS
    )
    np.savez_compressed(OOF_OUT, **oof_payload)
    report = {
        "method": VERSION,
        "status": "strict_renderer_passed" if passed else "analysis_only_no_candidate",
        "round": 2,
        "primary_hypothesis": (
            "Fingerprint subfamilies do not need to predict parent lifecycle: with parent KNN "
            "mass held exactly fixed, nearest-support family identity can select fit-only signed "
            "delay/H residuals and place the same mass more accurately in the physical renderer."
        ),
        "compliance": {
            "uses": [
                "official outer-fit H component fingerprints and signed assignment residuals",
                "official train coordinates",
                "official Round1_Map point-cloud source discovery inputs",
            ],
            "does_not_use": [
                "outer H in source discovery, KNN current, family selector, or renderer input",
                "v207/v212/synthetic truth or teacher",
                "restored RT channel/material coefficients",
                "local or online labels",
            ],
        },
        "parameters": {
            "parent_mass": "byte-equivalent v354 16-NN source mass",
            "family_selector": "nearest positive fit support within each parent source",
            "family_offsets": "fit-mass-weighted signed delay/H assignment residual mean",
            "transport_alpha": 1.0,
            "parameter_search": "none",
            "minimum_mean_weighted_delta_vs_v354": MIN_MEAN_DELTA_VS_V354,
            "minimum_fold_wins": MIN_FOLD_WINS,
        },
        "observation": observation_report,
        "fold_rows": fold_rows,
        "decision": {
            "mean_delta_vs_v354": mean_delta_vs_v354,
            "mean_delta_vs_shuffle": mean_delta_vs_shuffle,
            "mean_family_oracle_delta_vs_parent_oracle": mean_oracle_delta,
            "fold_wins": fold_wins,
            "passed": passed,
            "next_action": (
                "evaluate the passed family renderer in the strict final-current chain"
                if passed
                else "terminate fingerprint subfamily branch"
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
