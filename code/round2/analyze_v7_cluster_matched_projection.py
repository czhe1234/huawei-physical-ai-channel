#!/usr/bin/env python3
"""Audit cluster-matched V4 projection policies using only official Round 2 data."""

from __future__ import annotations

import json
import time

import numpy as np

from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import optimal_scale, scaled_error, statistics


VERSION = "Round2-E48-cluster-matched-projection-policy"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "cluster_matched_projection"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_cluster_matched_projection"
METRICS = (
    PATHS.artifacts
    / "round2"
    / "v4_calibrated_exact_projection"
    / "calibrated_exact_projection_metrics.npz"
)
V4_SELECTION = (
    PATHS.artifacts
    / "round2"
    / "v4_calibrated_exact_projection"
    / "calibrated_exact_projection_selection.json"
)
GEODESIC = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
REPORT = ANALYSIS_DIR / "cluster_matched_projection.json"
SELECTION = ARTIFACT_DIR / "cluster_matched_projection_selection.json"


def cluster_features(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    train_positions: np.ndarray,
    geometry: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    keys: list[tuple[int, int]] = []
    features: list[np.ndarray] = []
    for fold in np.unique(fold_id):
        for site_value in np.unique(site):
            rows = np.flatnonzero((fold_id == fold) & (site == site_value))
            if len(rows) == 0:
                continue
            point_ids = query_ids[rows]
            xy = train_positions[point_ids, :2]
            centered = xy - np.mean(xy, axis=0, keepdims=True)
            covariance = centered.T @ centered / max(len(xy) - 1, 1)
            eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
            spread = np.sqrt(np.maximum(eigenvalues, 0.0))
            geo = geometry[point_ids]
            feature = np.concatenate(
                [
                    np.asarray(
                        [
                            len(rows),
                            spread[0],
                            spread[1],
                            spread[0] / max(spread[1], 1e-4),
                        ],
                        dtype=np.float64,
                    ),
                    np.mean(geo, axis=0, dtype=np.float64),
                    np.std(geo, axis=0, dtype=np.float64),
                ]
            )
            keys.append((int(fold), int(site_value)))
            features.append(feature)
    return keys, np.stack(features)


def candidate_value(
    arrays: dict[str, np.ndarray],
    candidate: int,
    mask: np.ndarray,
) -> tuple[float, float, dict[str, float | int]]:
    scale = optimal_scale(
        arrays["predicted_energy"][candidate], arrays["cross"][candidate], mask
    )
    error = scaled_error(
        arrays["predicted_energy"][candidate],
        arrays["truth_energy"][candidate],
        arrays["cross"][candidate],
        scale,
    )
    value = statistics(
        arrays["pas"][candidate],
        arrays["pdp"][candidate],
        error,
        arrays["truth_energy"][candidate],
        mask,
    )
    return float(value["score"]), float(scale), value


def direct_cluster_choices(
    arrays: dict[str, np.ndarray],
    fold_id: np.ndarray,
    site: np.ndarray,
) -> dict[tuple[int, int], dict[str, object]]:
    names = arrays["candidate_names"]
    output: dict[tuple[int, int], dict[str, object]] = {}
    for fold in np.unique(fold_id):
        for site_value in np.unique(site):
            mask = (fold_id == fold) & (site == site_value)
            if not np.any(mask):
                continue
            best: dict[str, object] | None = None
            for candidate in range(len(names)):
                score, scale, value = candidate_value(arrays, candidate, mask)
                if best is None or score > float(best["score"]):
                    best = {
                        "fold": int(fold),
                        "site": int(site_value),
                        "candidate_index": int(candidate),
                        "candidate": str(names[candidate]),
                        "scale": scale,
                        "score": score,
                        "metrics": value,
                    }
            if best is None:
                raise RuntimeError("empty cluster candidate search")
            output[(int(fold), int(site_value))] = best
    return output


def apply_choices(
    arrays: dict[str, np.ndarray],
    fold_id: np.ndarray,
    site: np.ndarray,
    choices: dict[tuple[int, int], dict[str, object]],
) -> dict[str, object]:
    count = len(fold_id)
    pas = np.empty(count, dtype=np.float64)
    pdp = np.empty(count, dtype=np.float64)
    error = np.empty(count, dtype=np.float64)
    energy = np.empty(count, dtype=np.float64)
    for key, row in choices.items():
        fold, site_value = key
        mask = (fold_id == fold) & (site == site_value)
        candidate = int(row["candidate_index"])
        scale = float(row["scale"])
        pas[mask] = arrays["pas"][candidate, mask]
        pdp[mask] = arrays["pdp"][candidate, mask]
        energy[mask] = arrays["truth_energy"][candidate, mask]
        error[mask] = scaled_error(
            arrays["predicted_energy"][candidate, mask],
            arrays["truth_energy"][candidate, mask],
            arrays["cross"][candidate, mask],
            scale,
        )
    return {
        "overall": statistics(pas, pdp, error, energy),
        "folds": [
            {
                "fold": int(fold),
                **statistics(pas, pdp, error, energy, fold_id == fold),
            }
            for fold in np.unique(fold_id)
        ],
    }


def nearest_cluster_transfer(
    direct: dict[tuple[int, int], dict[str, object]],
    keys: list[tuple[int, int]],
    features: np.ndarray,
    arrays: dict[str, np.ndarray],
    fold_id: np.ndarray,
    site: np.ndarray,
) -> tuple[dict[tuple[int, int], dict[str, object]], list[dict[str, object]]]:
    output: dict[tuple[int, int], dict[str, object]] = {}
    records: list[dict[str, object]] = []
    for site_value in np.unique(site):
        local = np.asarray(
            [index for index, key in enumerate(keys) if key[1] == int(site_value)],
            dtype=np.int64,
        )
        mean = np.mean(features[local], axis=0, dtype=np.float64)
        scale = np.std(features[local], axis=0, dtype=np.float64)
        normalized = (features - mean) / np.maximum(scale, 1e-4)
        for destination in local:
            candidate_sources = local[local != destination]
            distance = np.sqrt(
                np.mean(
                    np.square(normalized[candidate_sources] - normalized[destination]),
                    axis=1,
                    dtype=np.float64,
                )
            )
            source_index = int(candidate_sources[int(np.argmin(distance))])
            source_key = keys[source_index]
            target_key = keys[destination]
            source = direct[source_key]
            mask = (fold_id == target_key[0]) & (site == target_key[1])
            candidate_index = int(source["candidate_index"])
            # Candidate identity transfers from the matched cluster. Its amplitude is
            # calibrated from all other clusters at the same site, excluding the target.
            fit = (fold_id != target_key[0]) & (site == target_key[1])
            scale_value = optimal_scale(
                arrays["predicted_energy"][candidate_index],
                arrays["cross"][candidate_index],
                fit,
            )
            score, _, value = candidate_value(
                arrays, candidate_index, mask
            )
            output[target_key] = {
                "fold": target_key[0],
                "site": target_key[1],
                "candidate_index": candidate_index,
                "candidate": str(arrays["candidate_names"][candidate_index]),
                "scale": float(scale_value),
            }
            records.append(
                {
                    "target": list(target_key),
                    "source": list(source_key),
                    "feature_distance": float(np.min(distance)),
                    "candidate": str(arrays["candidate_names"][candidate_index]),
                    "target_oracle_scaled_score_diagnostic": score,
                    "target_metrics_oracle_scaled_diagnostic": value,
                }
            )
    return output, records


def pointwise_outer_oracle(
    arrays: dict[str, np.ndarray],
    fold_id: np.ndarray,
    site: np.ndarray,
) -> dict[str, object]:
    count = len(fold_id)
    names = arrays["candidate_names"]
    selected = np.empty(count, dtype=np.int16)
    selected_scale = np.empty(count, dtype=np.float32)
    pas = np.empty(count, dtype=np.float64)
    pdp = np.empty(count, dtype=np.float64)
    error = np.empty(count, dtype=np.float64)
    energy = np.empty(count, dtype=np.float64)
    for fold in np.unique(fold_id):
        for site_value in np.unique(site):
            fit = (fold_id != fold) & (site == site_value)
            held_rows = np.flatnonzero((fold_id == fold) & (site == site_value))
            if len(held_rows) == 0:
                continue
            utility = np.empty((len(names), len(held_rows)), dtype=np.float64)
            scales = np.empty(len(names), dtype=np.float64)
            candidate_error = np.empty_like(utility)
            for candidate in range(len(names)):
                scale_value = optimal_scale(
                    arrays["predicted_energy"][candidate],
                    arrays["cross"][candidate],
                    fit,
                )
                scales[candidate] = scale_value
                current_error = scaled_error(
                    arrays["predicted_energy"][candidate, held_rows],
                    arrays["truth_energy"][candidate, held_rows],
                    arrays["cross"][candidate, held_rows],
                    scale_value,
                )
                candidate_error[candidate] = current_error
                local_energy = arrays["truth_energy"][candidate, held_rows]
                local_nmse = np.divide(
                    current_error,
                    np.maximum(local_energy, 1e-30),
                )
                utility[candidate] = (
                    0.4 * arrays["pas"][candidate, held_rows]
                    + 0.4 * arrays["pdp"][candidate, held_rows]
                    + 0.2 / (1.0 + local_nmse)
                )
            best = np.argmax(utility, axis=0)
            selected[held_rows] = best.astype(np.int16)
            selected_scale[held_rows] = scales[best].astype(np.float32)
            pas[held_rows] = arrays["pas"][best, held_rows]
            pdp[held_rows] = arrays["pdp"][best, held_rows]
            error[held_rows] = candidate_error[best, np.arange(len(held_rows))]
            energy[held_rows] = arrays["truth_energy"][best, held_rows]
    unique, counts = np.unique(selected, return_counts=True)
    ranking = sorted(
        (
            {
                "candidate_index": int(index),
                "candidate": str(names[index]),
                "count": int(value),
            }
            for index, value in zip(unique, counts)
        ),
        key=lambda row: int(row["count"]),
        reverse=True,
    )
    return {
        "overall": statistics(pas, pdp, error, energy),
        "folds": [
            {
                "fold": int(fold),
                **statistics(pas, pdp, error, energy, fold_id == fold),
            }
            for fold in np.unique(fold_id)
        ],
        "unique_candidates": int(len(unique)),
        "top_candidates": ranking[:20],
        "selected_candidate": selected.tolist(),
        "selected_scale": selected_scale.tolist(),
    }


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    loaded = np.load(METRICS, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    fold_id = arrays["fold_id"].astype(np.int8)
    site = arrays["site"].astype(np.int8)
    query_ids = arrays["query_ids"].astype(np.int64)
    _, train_positions, _, _, _ = load_positions()
    geometry = np.load(GEODESIC, allow_pickle=False)["train_features"].astype(
        np.float32
    )
    keys, features = cluster_features(
        query_ids, fold_id, site, train_positions, geometry
    )
    direct = direct_cluster_choices(arrays, fold_id, site)
    direct_score = apply_choices(arrays, fold_id, site, direct)
    point_oracle = pointwise_outer_oracle(arrays, fold_id, site)
    transferred, transfer_records = nearest_cluster_transfer(
        direct, keys, features, arrays, fold_id, site
    )
    transfer_score = apply_choices(arrays, fold_id, site, transferred)
    v4 = json.loads(V4_SELECTION.read_text(encoding="utf-8"))
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "official_inputs_only": True,
        "sionna_used": False,
        "primary_hypothesis": (
            "The validation holes were explicitly shaped after the ten test clusters; "
            "projection policy should therefore transfer by hole topology better than one "
            "site-wide policy."
        ),
        "v4_nested_reference": v4["nested_selected"],
        "v4_full_policy_reference": v4["final_policy_oof"]["overall"],
        "direct_cluster_oracle": {
            "score": direct_score,
            "choices": [direct[key] for key in keys],
            "interpretation": "upper bound only; each held cluster selected on its own truth",
        },
        "pointwise_outer_oracle": point_oracle,
        "nearest_other_cluster_transfer": {
            "score": transfer_score,
            "choices": [transferred[key] for key in keys],
            "records": transfer_records,
        },
        "cluster_keys": [list(key) for key in keys],
        "seconds": float(time.time() - started),
    }
    SELECTION.write_text(
        json.dumps(
            {
                "direct_cluster_choices": [direct[key] for key in keys],
                "transfer_choices": [transferred[key] for key in keys],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
