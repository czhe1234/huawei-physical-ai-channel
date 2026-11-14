#!/usr/bin/env python3
"""Build explicit local propagation tracks from delay and path fingerprints."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import analyze_v254_route_topology_spectral_field as v254
import analyze_v273_planar_barrier_kernel as v273
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v299_component_correspondence_oracle as v299
import analyze_v300_path_gain_fingerprint_oracle as v300
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 302
VERSION = "v302_fingerprint_local_tracks"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

OUTER_SEEDS = (302, 1302, 2302, 3302, 4302)
OUTER_COUNT = 320
INNER_COUNT = 240
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)
MAX_DELAY_GRADIENT = 0.03
AFFINE_RIDGE = 1e-3


@dataclass(frozen=True)
class Variant:
    name: str
    max_delay_span: float
    min_fingerprint_cosine: float
    minimum_persistence: int


VARIANTS = (
    Variant("tight_p2", 0.30, 0.78, 2),
    Variant("balanced_p2", 0.50, 0.72, 2),
    Variant("balanced_p3", 0.50, 0.72, 3),
    Variant("wide_p3", 0.75, 0.66, 3),
    Variant("wide_p4", 0.75, 0.66, 4),
)


def combined_similarity(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    polarization: np.ndarray,
    ue_port: np.ndarray,
) -> np.ndarray:
    def matrix(value: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(value, axis=1)
        return value @ value.T / np.maximum(norm[:, None] * norm[None, :], 1e-30)

    return (
        0.55 * matrix(horizontal)
        + 0.20 * matrix(vertical)
        + 0.10 * matrix(polarization)
        + 0.15 * matrix(ue_port)
    )


def circular_center(delay: np.ndarray, weight: np.ndarray) -> float:
    angle = 2.0 * np.pi * delay / v297.N_DELAY
    vector = np.sum(weight * np.exp(1j * angle))
    return float((np.angle(vector) % (2.0 * np.pi)) * v297.N_DELAY / (2.0 * np.pi))


def cluster_labels(
    delay: np.ndarray,
    owner: np.ndarray,
    fingerprint_similarity: np.ndarray,
    variant: Variant,
) -> np.ndarray:
    if len(delay) == 1:
        return np.ones(1, dtype=np.int32)
    delay_distance = np.abs(
        (delay[:, None] - delay[None, :] + v297.N_DELAY / 2.0)
        % v297.N_DELAY
        - v297.N_DELAY / 2.0
    )
    fingerprint_distance = np.maximum(1.0 - fingerprint_similarity, 0.0)
    distance = np.maximum(
        delay_distance / variant.max_delay_span,
        fingerprint_distance / max(1.0 - variant.min_fingerprint_cosine, 1e-6),
    )
    distance[owner[:, None] == owner[None, :]] = 10.0
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="complete", optimal_ordering=False)
    return fcluster(tree, t=1.0, criterion="distance").astype(np.int32)


def track_prediction(
    query_id: int,
    neighbor_ids: np.ndarray,
    base_weights: np.ndarray,
    all_pos: np.ndarray,
    component: dict[str, np.ndarray],
    fingerprint: dict[str, np.ndarray],
    variant: Variant,
) -> tuple[np.ndarray, dict[str, float]]:
    valid = component["valid"][neighbor_ids]
    if not np.any(valid):
        return np.full(v297.N_DELAY, 1.0 / v297.N_DELAY, dtype=np.float32), {
            "track_count": 0.0,
            "retained_mass": 0.0,
            "mean_persistence": 0.0,
        }
    owner = np.broadcast_to(np.arange(len(neighbor_ids))[:, None], valid.shape)[valid]
    point_ids = np.broadcast_to(neighbor_ids[:, None], valid.shape)[valid]
    component_ids = np.broadcast_to(
        np.arange(valid.shape[1])[None, :], valid.shape
    )[valid]
    delay = component["delay"][neighbor_ids][valid].astype(np.float64)
    token_mass = (
        base_weights[:, None] * component["weight"][neighbor_ids]
    )[valid].astype(np.float64)
    h = fingerprint["horizontal"][point_ids, component_ids].astype(np.float64)
    v = fingerprint["vertical"][point_ids, component_ids].astype(np.float64)
    p = fingerprint["polarization"][point_ids, component_ids].astype(np.float64)
    u = fingerprint["ue_port"][point_ids, component_ids].astype(np.float64)
    similarity = combined_similarity(h, v, p, u)
    labels = cluster_labels(delay, owner, similarity, variant)

    track_delay = []
    track_mass = []
    persistence_values = []
    for label in np.unique(labels):
        selected = labels == label
        persistence = len(np.unique(owner[selected]))
        if persistence < variant.minimum_persistence:
            continue
        weight = token_mass[selected]
        if float(np.sum(weight)) <= 1e-20:
            continue
        weight /= np.sum(weight)
        reference = circular_center(delay[selected], weight)
        xy = all_pos[point_ids[selected], :2] - all_pos[query_id, :2]
        prediction = v299.fit_affine_at_query(
            xy, delay[selected], weight, reference
        ) if len(weight) >= 3 else reference
        track_delay.append(prediction)
        track_mass.append(float(np.sum(token_mass[selected])))
        persistence_values.append(float(persistence))

    diffuse = float(np.sum(base_weights * component["diffuse"][neighbor_ids]))
    if not track_delay:
        return np.full(v297.N_DELAY, 1.0 / v297.N_DELAY, dtype=np.float32), {
            "track_count": 0.0,
            "retained_mass": diffuse,
            "mean_persistence": 0.0,
        }
    atoms = v297.dirichlet_atoms(np.asarray(track_delay, dtype=np.float32))
    mass = np.asarray(track_mass, dtype=np.float32)
    prediction = atoms @ mass + diffuse / v297.N_DELAY
    prediction /= np.maximum(np.sum(prediction), 1e-20)
    return prediction.astype(np.float32), {
        "track_count": float(len(track_delay)),
        "retained_mass": float(np.sum(track_mass) + diffuse),
        "mean_persistence": float(np.mean(persistence_values)),
    }


def predict_set(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    all_pos: np.ndarray,
    signatures: np.ndarray,
    blocker_count: np.ndarray,
    footprint: np.ndarray,
    z_max: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    truth_d: np.ndarray,
    component: dict[str, np.ndarray],
    fingerprint: dict[str, np.ndarray],
    variant: Variant,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    raw = v274.build_raw_set(
        query_ids,
        bank_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )
    base = v297.normalize_rows(
        np.sum(
            raw.base_weights[:, :, None] * truth_d[raw.neighbor_ids],
            axis=1,
            dtype=np.float64,
        )
    )
    model = np.empty_like(base)
    diagnostics = {
        "track_count": np.zeros(len(query_ids), dtype=np.float32),
        "retained_mass": np.zeros(len(query_ids), dtype=np.float32),
        "mean_persistence": np.zeros(len(query_ids), dtype=np.float32),
    }
    for row, query_id in enumerate(query_ids):
        prediction, current = track_prediction(
            int(query_id),
            raw.neighbor_ids[row],
            raw.base_weights[row],
            all_pos,
            component,
            fingerprint,
            variant,
        )
        model[row] = prediction
        for key in diagnostics:
            diagnostics[key][row] = current[key]
    return base, model, diagnostics


def choose_variant(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    geometry: tuple,
    truth_d: np.ndarray,
    component: dict[str, np.ndarray],
    fingerprint: dict[str, np.ndarray],
) -> tuple[Variant, float, list[dict[str, object]]]:
    rows = []
    for variant in VARIANTS:
        base, model, diagnostics = predict_set(
            query_ids, bank_ids, *geometry, truth_d, component, fingerprint, variant
        )
        truth = truth_d[query_ids]
        base_score = float(np.mean(v297.row_cos(base, truth)))
        best = None
        for alpha in ALPHA_GRID:
            blended = v297.normalize_rows(base + float(alpha) * (model - base))
            score = float(np.mean(v297.row_cos(blended, truth)))
            item = {"alpha": float(alpha), "pdp": score, "delta": score - base_score}
            if best is None or score > best["pdp"]:
                best = item
        assert best is not None
        rows.append(
            {
                "variant": variant.name,
                "base_pdp": base_score,
                "best": best,
                "diagnostics": {
                    key: summarize(value) for key, value in diagnostics.items()
                },
            }
        )
    selected = max(rows, key=lambda row: row["best"]["pdp"])
    variant = next(value for value in VARIANTS if value.name == selected["variant"])
    return variant, float(selected["best"]["alpha"]), rows


def main() -> None:
    if not v297.COMPONENT_CACHE.is_file() or not v300.FINGERPRINT_CACHE.is_file():
        raise FileNotFoundError("v297 component or v300 fingerprint cache")
    component_data = np.load(v297.COMPONENT_CACHE)
    component = {key: component_data[key] for key in component_data.files}
    fingerprint_data = np.load(v300.FINGERPRINT_CACHE)
    fingerprint = {key: fingerprint_data[key] for key in fingerprint_data.files}
    targets = np.load(v297.TARGET_CACHE)
    truth_d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    (
        all_pos,
        signatures,
        blocker_count,
        _scene_features,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = v274.load_geometry()
    geometry = (
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )

    fold_rows = []
    base_parts = []
    prediction_parts = []
    truth_parts = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = v273.matched_subset(
            outer_fit,
            train_pos[:, :2],
            test_pos[:, :2],
            INNER_COUNT,
            seed + 10_000,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        variant, alpha, search = choose_variant(
            inner_ids,
            inner_fit,
            geometry,
            truth_d,
            component,
            fingerprint,
        )
        base, model, diagnostics = predict_set(
            outer_ids,
            outer_fit,
            *geometry,
            truth_d,
            component,
            fingerprint,
            variant,
        )
        prediction = v297.normalize_rows(base + alpha * (model - base))
        truth = truth_d[outer_ids]
        base_score = float(np.mean(v297.row_cos(base, truth)))
        score = float(np.mean(v297.row_cos(prediction, truth)))
        base_parts.append(base)
        prediction_parts.append(prediction)
        truth_parts.append(truth)
        fold_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "selected_variant": variant.name,
                "selected_alpha": alpha,
                "base_pdp": base_score,
                "track_pdp": score,
                "delta": score - base_score,
                "inner_search": search,
                "outer_diagnostics": {
                    key: summarize(value) for key, value in diagnostics.items()
                },
            }
        )
        print(
            f"v302 fold {fold + 1}/5 variant={variant.name} alpha={alpha:.3f} "
            f"base={base_score:.6f} track={score:.6f} delta={score-base_score:+.6f}",
            flush=True,
        )

    base_all = np.concatenate(base_parts)
    prediction_all = np.concatenate(prediction_parts)
    truth_all = np.concatenate(truth_parts)
    base_score = float(np.mean(v297.row_cos(base_all, truth_all)))
    track_score = float(np.mean(v297.row_cos(prediction_all, truth_all)))
    fold_deltas = [float(row["delta"]) for row in fold_rows]
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Explicit complete-link tracks in continuous-delay and path-fingerprint "
            "space resolve the family ambiguity that free token attention did not learn."
        ),
        "variants": [variant.__dict__ for variant in VARIANTS],
        "fold_rows": fold_rows,
        "oof": {
            "base_pdp": base_score,
            "track_pdp": track_score,
            "delta": track_score - base_score,
            "fold_deltas": fold_deltas,
            "fold_wins": int(sum(value > 0.0 for value in fold_deltas)),
        },
        "compliance_note": (
            "This deterministic track predictor is diagnostic only. Even if positive, "
            "a submission must use it as a physical feature inside a trained AI gate."
        ),
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["oof"], indent=2), flush=True)


if __name__ == "__main__":
    main()
