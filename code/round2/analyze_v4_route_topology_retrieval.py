#!/usr/bin/env python3
"""Audit route-topology donor retrieval on strict Round 2 spatial holes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from data import load_positions
from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "route_topology"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_route_topology"
ROUTE_CACHE = ARTIFACT_DIR / "round2_route_topology_features.npz"
REPORT = ANALYSIS_DIR / "route_topology_retrieval.json"
PREDICTIONS = ARTIFACT_DIR / "route_topology_retrieval_oof.npz"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
GEOMETRY = PRE_DIR / "round2_geometry_visibility_features.npz"
OOF_META = PRE_DIR / "round2_pre_sionna_oof_foundation_meta.npz"
FOLD_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
V2_WEIGHTS = (
    PATHS.artifacts / "round2" / "rbf_spectral_field" / "round2_rbf_oof_weights.npz"
)

CANDIDATES = 192
DONOR_COUNTS = (4, 8, 16, 32)
ROUTE_STRENGTHS = (0.25, 0.50, 1.00, 2.00, 4.00)
BLEND_ALPHAS = (0.25, 0.50, 0.75, 1.00)


@dataclass(frozen=True)
class Config:
    mode: str
    donors: int
    route_strength: float
    alpha: float

    @property
    def name(self) -> str:
        return (
            f"{self.mode}_k{self.donors}_r{self.route_strength:g}_a{self.alpha:g}"
        )


def normalize_power(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = output.sum(axis=1, keepdims=True)
    return np.divide(
        output,
        np.maximum(total, 1e-30),
        out=np.zeros_like(output),
        where=total > 0.0,
    )


def normalize_amplitude(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    norm = np.linalg.norm(output, axis=1, keepdims=True)
    return np.divide(
        output,
        np.maximum(norm, 1e-30),
        out=np.zeros_like(output),
        where=norm > 0.0,
    )


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        np.maximum(denominator, 1e-30),
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def route_signature(cache: np.lib.npyio.NpzFile, prefix: str) -> dict[str, np.ndarray]:
    return {
        "hit": cache[f"{prefix}_route_hit"].astype(np.float32),
        "occupancy": cache[f"{prefix}_route_occupancy"].astype(np.float32),
        "count": cache[f"{prefix}_route_count"].astype(np.float32),
        "panorama": cache[f"{prefix}_panorama"].astype(np.float32),
    }


def route_distance(
    signature: dict[str, np.ndarray],
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
) -> np.ndarray:
    query_hit = signature["hit"][query_ids, None]
    donor_hit = signature["hit"][donor_ids]
    hit = np.mean(np.abs(query_hit - donor_hit), axis=(2, 3))
    query_occupancy = signature["occupancy"][query_ids, None]
    donor_occupancy = signature["occupancy"][donor_ids]
    occupancy = np.mean(np.abs(query_occupancy - donor_occupancy), axis=(2, 3))
    count = np.mean(
        np.abs(signature["count"][query_ids, None] - signature["count"][donor_ids])
        / 8.0,
        axis=2,
    )
    panorama = np.mean(
        np.abs(
            signature["panorama"][query_ids, None]
            - signature["panorama"][donor_ids]
        ),
        axis=(2, 3),
    )
    return occupancy + 0.55 * hit + 0.35 * count + 0.75 * panorama


def geometry_distance(
    geometry: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
) -> np.ndarray:
    selected = geometry[:, 6:].astype(np.float64)
    mean = np.mean(selected[fit_ids], axis=0)
    std = np.maximum(np.std(selected[fit_ids], axis=0), 1e-4)
    standardized = np.clip((selected - mean) / std, -8.0, 8.0)
    difference = standardized[query_ids, None] - standardized[donor_ids]
    return np.sqrt(np.mean(difference * difference, axis=2))


def predict_amplitude(
    label: np.ndarray,
    donor_ids: np.ndarray,
    spatial_distance: np.ndarray,
    state_distance: np.ndarray,
    donors: int,
    strength: float,
) -> np.ndarray:
    local_scale = np.maximum(spatial_distance[:, :8].mean(axis=1, keepdims=True), 1.0)
    score = spatial_distance / local_scale + strength * state_distance
    selected = np.argpartition(score, donors - 1, axis=1)[:, :donors]
    selected_score = np.take_along_axis(score, selected, axis=1)
    selected_ids = np.take_along_axis(donor_ids, selected, axis=1)
    selected_score -= np.min(selected_score, axis=1, keepdims=True)
    weight = np.exp(-selected_score)
    weight /= np.maximum(np.sum(weight, axis=1, keepdims=True), 1e-30)
    amplitude = np.einsum(
        "qk,qkd->qd", weight, label[selected_ids], optimize=True
    )
    return normalize_amplitude(amplitude)


def main() -> None:
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (ROUTE_CACHE, DESCRIPTORS, GEOMETRY, OOF_META, FOLD_CACHE, V2_WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(path)

    _, positions, _, site, _ = load_positions()
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    pas_truth = normalize_power(descriptors["global_pas"])
    pdp_truth = normalize_power(descriptors["global_pdp"])
    pas_amplitude = np.sqrt(pas_truth)
    pdp_amplitude = np.sqrt(pdp_truth)
    geometry = np.load(GEOMETRY)["train"].astype(np.float32)
    route = route_signature(np.load(ROUTE_CACHE), "train")
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    query_site = site[query_ids]
    folds = np.load(FOLD_CACHE)
    v2 = np.load(V2_WEIGHTS)

    base_pas_amplitude = np.empty((len(query_ids), pas_amplitude.shape[1]))
    base_pdp_amplitude = np.empty((len(query_ids), pdp_amplitude.shape[1]))
    for row, (ids, weight) in enumerate(zip(v2["neighbor_ids"], v2["weights"])):
        base_pas_amplitude[row] = normalize_amplitude(
            np.sum(weight[:, None] * pas_amplitude[ids], axis=0)[None]
        )[0]
        base_pdp_amplitude[row] = normalize_amplitude(
            np.sum(weight[:, None] * pdp_amplitude[ids], axis=0)[None]
        )[0]
    base_pas_score = row_cosine(base_pas_amplitude**2, pas_truth[query_ids])
    base_pdp_score = row_cosine(base_pdp_amplitude**2, pdp_truth[query_ids])
    base_point = 0.5 * (base_pas_score + base_pdp_score)

    configs = [
        Config(mode, donors, strength, alpha)
        for mode in ("route", "geometry", "shuffled_route")
        for donors in DONOR_COUNTS
        for strength in ROUTE_STRENGTHS
        for alpha in BLEND_ALPHAS
    ]
    predicted_pas = {config.name: np.empty_like(base_pas_amplitude) for config in configs}
    predicted_pdp = {config.name: np.empty_like(base_pdp_amplitude) for config in configs}
    rng = np.random.default_rng(20_260_805)
    permutation = np.arange(len(positions), dtype=np.int64)
    for site_value in np.unique(site):
        rows = np.flatnonzero(site == site_value)
        permutation[rows] = rng.permutation(rows)
    shuffled = {key: value[permutation] for key, value in route.items()}

    for fold in np.unique(fold_id):
        fold_rows = np.flatnonzero(fold_id == fold)
        fit_all = folds[f"fold{int(fold)}_fit"].astype(np.int64)
        for site_value in np.unique(query_site[fold_rows]):
            output_rows = fold_rows[query_site[fold_rows] == site_value]
            query = query_ids[output_rows]
            fit = fit_all[(site[fit_all] == site_value) & valid[fit_all]]
            spatial, local = cKDTree(positions[fit, :2]).query(
                positions[query, :2], k=min(CANDIDATES, len(fit)), workers=-1
            )
            donor = fit[local]
            distances = {
                "route": route_distance(route, query, donor),
                "geometry": geometry_distance(geometry, fit, query, donor),
                "shuffled_route": route_distance(shuffled, query, donor),
            }
            for mode, state_distance in distances.items():
                median = np.median(state_distance, axis=1, keepdims=True)
                state_distance = state_distance / np.maximum(median, 1e-6)
                for donors in DONOR_COUNTS:
                    for strength in ROUTE_STRENGTHS:
                        proposal_pas = predict_amplitude(
                            pas_amplitude, donor, spatial, state_distance, donors, strength
                        )
                        proposal_pdp = predict_amplitude(
                            pdp_amplitude, donor, spatial, state_distance, donors, strength
                        )
                        for alpha in BLEND_ALPHAS:
                            config = Config(mode, donors, strength, alpha)
                            predicted_pas[config.name][output_rows] = normalize_amplitude(
                                (1.0 - alpha) * base_pas_amplitude[output_rows]
                                + alpha * proposal_pas
                            )
                            predicted_pdp[config.name][output_rows] = normalize_amplitude(
                                (1.0 - alpha) * base_pdp_amplitude[output_rows]
                                + alpha * proposal_pdp
                            )
            print(
                f"route retrieval fold={int(fold)} site={int(site_value)} "
                f"query={len(query)} fit={len(fit)}",
                flush=True,
            )

    rows: list[dict[str, object]] = []
    point_score: dict[str, np.ndarray] = {}
    for config in configs:
        pas_score = row_cosine(predicted_pas[config.name] ** 2, pas_truth[query_ids])
        pdp_score = row_cosine(predicted_pdp[config.name] ** 2, pdp_truth[query_ids])
        joint = 0.5 * (pas_score + pdp_score)
        point_score[config.name] = joint
        rows.append(
            {
                "name": config.name,
                "config": config.__dict__,
                "pas": float(np.mean(pas_score)),
                "pdp": float(np.mean(pdp_score)),
                "mean": float(np.mean(joint)),
                "delta": float(np.mean(joint - base_point)),
                "fold_delta": [
                    float(np.mean(joint[fold_id == fold] - base_point[fold_id == fold]))
                    for fold in np.unique(fold_id)
                ],
            }
        )

    nested_pas = np.empty_like(base_pas_amplitude)
    nested_pdp = np.empty_like(base_pdp_amplitude)
    nested_choices = []
    eligible = [row for row in rows if row["config"]["mode"] != "shuffled_route"]
    for held_fold in np.unique(fold_id):
        train_mask = fold_id != held_fold
        held_mask = fold_id == held_fold
        selected = max(
            eligible,
            key=lambda row: float(np.mean(point_score[row["name"]][train_mask])),
        )
        nested_pas[held_mask] = predicted_pas[selected["name"]][held_mask]
        nested_pdp[held_mask] = predicted_pdp[selected["name"]][held_mask]
        nested_choices.append(
            {
                "held_fold": int(held_fold),
                "config": selected["config"],
                "held_delta": float(
                    np.mean(point_score[selected["name"]][held_mask] - base_point[held_mask])
                ),
            }
        )
    nested_pas_score = row_cosine(nested_pas**2, pas_truth[query_ids])
    nested_pdp_score = row_cosine(nested_pdp**2, pdp_truth[query_ids])
    nested_point = 0.5 * (nested_pas_score + nested_pdp_score)

    best_route = max(
        [row for row in rows if row["config"]["mode"] == "route"],
        key=lambda row: row["mean"],
    )
    matched_geometry = next(
        row
        for row in rows
        if row["config"] == {**best_route["config"], "mode": "geometry"}
    )
    matched_shuffle = next(
        row
        for row in rows
        if row["config"] == {**best_route["config"], "mode": "shuffled_route"}
    )
    ranked = sorted(rows, key=lambda row: row["mean"], reverse=True)
    np.savez_compressed(
        PREDICTIONS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        base_pas=base_pas_amplitude.astype(np.float16),
        base_pdp=base_pdp_amplitude.astype(np.float16),
        nested_pas=nested_pas.astype(np.float16),
        nested_pdp=nested_pdp.astype(np.float16),
    )
    report = {
        "experiment": "Round2-E11-route-topology-retrieval",
        "status": "unsupervised_retrieval_complete",
        "primary_hypothesis": (
            "High-resolution BS-to-UE obstruction sequences and local wall panoramas "
            "identify propagation-compatible donors beyond XY and aggregate geometry."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "baseline": {
            "pas": float(np.mean(base_pas_score)),
            "pdp": float(np.mean(base_pdp_score)),
            "mean": float(np.mean(base_point)),
        },
        "nested": {
            "pas": float(np.mean(nested_pas_score)),
            "pdp": float(np.mean(nested_pdp_score)),
            "mean": float(np.mean(nested_point)),
            "delta": float(np.mean(nested_point - base_point)),
            "fold_delta": [
                float(np.mean(nested_point[fold_id == fold] - base_point[fold_id == fold]))
                for fold in np.unique(fold_id)
            ],
            "choices": nested_choices,
        },
        "best_route": best_route,
        "matched_controls": {
            "geometry": matched_geometry,
            "shuffled_route": matched_shuffle,
            "route_vs_geometry": float(best_route["mean"] - matched_geometry["mean"]),
            "route_vs_shuffled": float(best_route["mean"] - matched_shuffle["mean"]),
        },
        "top_ten": ranked[:10],
        "outputs": {
            "predictions": PATHS.relative(PREDICTIONS),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
