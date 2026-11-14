#!/usr/bin/env python3
"""Audit independent candidate-lifecycle information beyond the E125 identity core."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e123_complete_path_variable_system as e123
import analyze_e124_one_to_one_path_identity_ceiling as e124
import analyze_global_local_path_lifecycle as e79
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
import train_e125_identity_first_path_operator as e125
import train_e126_conditional_path_state_heads as e126
from common.official_metrics import PointMetricStatistics, aggregate_statistics
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E127-candidate-lifecycle-information"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "candidate_lifecycle_information"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_candidate_lifecycle_information"
REPORT = ANALYSIS_DIR / "candidate_lifecycle_outer0_audit.json"
METRICS = ARTIFACT_DIR / "candidate_lifecycle_outer0_metrics.npz"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
OUTER = 0
INNER = 1
NEIGHBORS = 8
SPATIAL_SCALE_M = 12.0
MATCH_BANDWIDTH = 0.25
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_808_127)
    return parser.parse_args()


@torch.inference_mode()
def identity_predictions(
    model: e118.HierarchicalUniquePathOperator,
    store: e118.ClusterTensorStore,
    normalization: dict[str, torch.Tensor],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    logits_parts: list[np.ndarray] = []
    mass_parts: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(store.mask), batch_size):
        stop = min(start + batch_size, len(store.mask))
        rows = torch.arange(start, stop, device=store.device)
        output = model(store.batch(rows, normalization))
        logits_parts.append(output["mass_logits"].cpu().numpy())
        mass_parts.append(output["mass"].cpu().numpy())
    return np.concatenate(logits_parts), np.concatenate(mass_parts)


def build_neighbors(
    positions: np.ndarray,
    site: np.ndarray,
    fold_id: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(positions)
    neighbors = np.full((count, NEIGHBORS), -1, dtype=np.int32)
    shuffled = np.full((count, NEIGHBORS), -1, dtype=np.int32)
    distance = np.full((count, NEIGHBORS), np.inf, dtype=np.float32)
    rng = np.random.default_rng(seed)
    for fold in range(5):
        for site_value in np.unique(site):
            rows = np.flatnonzero((fold_id == fold) & (site == site_value))
            if len(rows) <= 2 * NEIGHBORS + 1:
                raise RuntimeError(
                    f"insufficient lifecycle rows fold={fold} site={site_value}"
                )
            local_distance, local_index = cKDTree(positions[rows, :2]).query(
                positions[rows, :2], k=NEIGHBORS + 1, workers=-1
            )
            neighbors[rows] = rows[np.asarray(local_index)[:, 1:]].astype(np.int32)
            distance[rows] = np.asarray(local_distance, np.float32)[:, 1:]
            for row in rows:
                forbidden = np.concatenate(([row], neighbors[row]))
                pool = rows[~np.isin(rows, forbidden)]
                shuffled[row] = rng.choice(pool, size=NEIGHBORS, replace=False)
    if np.any(neighbors < 0) or np.any(shuffled < 0):
        raise RuntimeError("incomplete lifecycle neighbor table")
    row_ids = np.arange(count)[:, None]
    if np.any(shuffled == row_ids):
        raise RuntimeError("shuffled lifecycle control contains query self-evidence")
    if any(set(neighbors[row]) & set(shuffled[row]) for row in range(count)):
        raise RuntimeError("shuffled lifecycle control overlaps correct neighbors")
    return neighbors, shuffled, distance


def lifecycle_features(
    coordinate: np.ndarray,
    mask: np.ndarray,
    source_mass: np.ndarray,
    source_probability: np.ndarray,
    neighbor_ids: np.ndarray,
    neighbor_distance: np.ndarray,
) -> np.ndarray:
    count, candidates = mask.shape
    output = np.zeros((count, candidates, 8), dtype=np.float32)
    for row in range(count):
        query_ids = np.flatnonzero(mask[row])
        if len(query_ids) == 0:
            continue
        spatial_weight = np.exp(-neighbor_distance[row] / SPATIAL_SCALE_M)
        spatial_weight /= max(float(spatial_weight.sum()), TINY)
        matched_mass = np.zeros((NEIGHBORS, len(query_ids)), dtype=np.float32)
        matched_probability = np.zeros_like(matched_mass)
        closeness = np.zeros_like(matched_mass)
        minimum_cost = np.full_like(matched_mass, 20.0)
        for rank, neighbor in enumerate(neighbor_ids[row]):
            neighbor_valid = np.flatnonzero(mask[neighbor])
            if len(neighbor_valid) == 0:
                continue
            cost = e79.support_cost(
                coordinate[row, query_ids], coordinate[neighbor, neighbor_valid]
            )
            nearest_local = np.argmin(cost, axis=1)
            nearest = neighbor_valid[nearest_local]
            local_cost = cost[np.arange(len(query_ids)), nearest_local]
            kernel = np.exp(-0.5 * local_cost / MATCH_BANDWIDTH)
            minimum_cost[rank] = local_cost
            closeness[rank] = kernel
            matched_mass[rank] = source_mass[neighbor, nearest] * kernel
            matched_probability[rank] = source_probability[neighbor, nearest] * kernel
        weighted_mass = spatial_weight[:, None] * matched_mass
        weighted_probability = spatial_weight[:, None] * matched_probability
        weighted_closeness = spatial_weight[:, None] * closeness
        mean_mass = weighted_mass.sum(axis=0)
        mean_probability = weighted_probability.sum(axis=0)
        variance_mass = (
            spatial_weight[:, None]
            * np.square(matched_mass - mean_mass[None])
        ).sum(axis=0)
        output[row, query_ids, 0] = mean_mass
        output[row, query_ids, 1] = matched_mass.max(axis=0)
        output[row, query_ids, 2] = np.sqrt(np.maximum(variance_mass, 0.0))
        output[row, query_ids, 3] = mean_probability
        output[row, query_ids, 4] = weighted_closeness.sum(axis=0)
        output[row, query_ids, 5] = (
            spatial_weight[:, None] * (minimum_cost <= MATCH_BANDWIDTH)
        ).sum(axis=0)
        output[row, query_ids, 6] = (
            spatial_weight[:, None] * np.minimum(minimum_cost, 20.0)
        ).sum(axis=0)
        output[row, query_ids, 7] = minimum_cost.min(axis=0)
    return output


def unary_features(
    identity_logits: np.ndarray,
    identity_mass: np.ndarray,
    base_log_mass: np.ndarray,
    member_count: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    count, candidates = mask.shape
    entropy = -np.sum(
        identity_mass * np.log(np.maximum(identity_mass, 1e-8)), axis=1
    )
    rank = np.broadcast_to(
        np.arange(candidates, dtype=np.float32)[None] / max(candidates - 1, 1),
        (count, candidates),
    )
    return np.stack(
        (
            np.clip(identity_logits, -20.0, 20.0),
            np.log(np.maximum(identity_mass, 1e-8)),
            np.clip(base_log_mass, -20.0, 20.0),
            np.log1p(member_count.astype(np.float32)),
            rank,
            np.broadcast_to(entropy[:, None], (count, candidates)),
        ),
        axis=2,
    ).astype(np.float32)


def fit_probe(
    features: np.ndarray,
    target_active: np.ndarray,
    mask: np.ndarray,
    fit_rows: np.ndarray,
) -> Pipeline:
    valid = mask[fit_rows]
    x = features[fit_rows][valid]
    y = target_active[fit_rows][valid].astype(np.int8)
    model = Pipeline(
        (
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=500,
                    solver="lbfgs",
                    random_state=0,
                ),
            ),
        )
    )
    model.fit(x, y)
    return model


def probe_scores(
    model: Pipeline,
    features: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    output = np.full(mask.shape, -30.0, dtype=np.float32)
    output[mask] = model.decision_function(features[mask]).astype(np.float32)
    return output


def probe_payload(model: Pipeline) -> dict[str, object]:
    scaler: StandardScaler = model.named_steps["scale"]
    logistic: LogisticRegression = model.named_steps["logistic"]
    return {
        "standardizer_mean": scaler.mean_.tolist(),
        "standardizer_scale": scaler.scale_.tolist(),
        "coefficient": logistic.coef_[0].tolist(),
        "intercept": float(logistic.intercept_[0]),
    }


def centered_score(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, score, -1e4).astype(np.float64)
    maximum = masked.max(axis=1, keepdims=True)
    log_total = maximum + np.log(
        np.exp(masked - maximum).sum(axis=1, keepdims=True)
    )
    return np.where(mask, masked - log_total, -1e4).astype(np.float32)


def blend_score(
    identity_logits: np.ndarray,
    probe_logit: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    identity = centered_score(identity_logits, mask)
    probe = centered_score(probe_logit, mask)
    return np.where(mask, (1.0 - alpha) * identity + alpha * probe, -1e4).astype(
        np.float32
    )


@torch.inference_mode()
def evaluate_score(
    score: np.ndarray,
    store: e118.ClusterTensorStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    hard_mass: torch.Tensor,
    batch_size: int,
    include_complex: bool,
) -> dict[str, object]:
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r") if include_complex else None
    logits_parts: list[np.ndarray] = []
    positive_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    affinity_parts: list[np.ndarray] = []
    top16_parts: list[np.ndarray] = []
    top24_parts: list[np.ndarray] = []
    pas_parts: list[np.ndarray] = []
    pdp_parts: list[np.ndarray] = []
    complex_pas_parts: list[np.ndarray] = []
    complex_pdp_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        index = torch.as_tensor(local, dtype=torch.long, device=store.device)
        batch = e125.hard_batch(store, index, normalization, hard_mass)
        local_score = torch.as_tensor(score[local], device=store.device)
        local_score = local_score.masked_fill(~batch["mask"], -1e4)
        output = e125.base_output(batch)
        output["mass_logits"] = local_score
        output["mass"] = torch.softmax(local_score, dim=1)
        target_absolute = batch["target_mass"]
        target_total = target_absolute.sum(dim=1)
        nonzero = target_total > 0.0
        target = target_absolute / target_total[:, None].clamp_min(TINY)
        affinity = (
            torch.sqrt(output["mass"].clamp_min(1e-12))
            * torch.sqrt(target.clamp_min(0.0))
        ).sum(dim=1)
        affinity_parts.append(affinity[nonzero].cpu().numpy())
        top16_parts.append(
            e125.top_mass_recall(output["mass"], target, 16)[nonzero].cpu().numpy()
        )
        top24_parts.append(
            e125.top_mass_recall(output["mass"], target, 24)[nonzero].cpu().numpy()
        )
        logits_parts.append(local_score.cpu().numpy())
        positive_parts.append((target_absolute > 0.0).cpu().numpy())
        valid_parts.append(batch["mask"].cpu().numpy())
        pas, pdp, _, _ = e115.sufficient_scores(output, batch)
        pas_parts.append(pas.cpu().numpy())
        pdp_parts.append(pdp.cpu().numpy())
        if include_complex:
            prediction = e115.render_complex_channel(output)
            truth = torch.as_tensor(
                np.asarray(train_h[local], dtype=np.complex64), device=store.device
            )
            pas_full, pdp_full, error, energy = e115.torch_official_metrics(
                prediction, truth
            )
            complex_pas_parts.append(pas_full.cpu().numpy())
            complex_pdp_parts.append(pdp_full.cpu().numpy())
            error_parts.append(error.cpu().numpy())
            energy_parts.append(energy.cpu().numpy())
    logits = np.concatenate(logits_parts)
    positive = np.concatenate(positive_parts)
    valid = np.concatenate(valid_parts)
    result: dict[str, object] = {
        "count": int(len(rows)),
        "existence_ap": float(
            average_precision_score(positive[valid].astype(np.int8), logits[valid])
        ),
        "mass_bhattacharyya": float(np.mean(np.concatenate(affinity_parts))),
        "top16_target_mass_recall": float(np.mean(np.concatenate(top16_parts))),
        "top24_target_mass_recall": float(np.mean(np.concatenate(top24_parts))),
        "endpoint_pas": float(np.mean(np.concatenate(pas_parts))),
        "endpoint_pdp": float(np.mean(np.concatenate(pdp_parts))),
    }
    result["selection_score"] = float(
        0.4 * result["endpoint_pas"]
        + 0.4 * result["endpoint_pdp"]
        + 0.2 * result["mass_bhattacharyya"]
    )
    if include_complex:
        statistics = PointMetricStatistics(
            pas=np.concatenate(complex_pas_parts),
            pdp=np.concatenate(complex_pdp_parts),
            squared_error=np.concatenate(error_parts),
            truth_energy=np.concatenate(energy_parts),
        )
        result["full_official"] = aggregate_statistics(statistics)
    return result


def select_alpha(
    name: str,
    probe_logit: np.ndarray,
    identity_logits: np.ndarray,
    mask: np.ndarray,
    store: e118.ClusterTensorStore,
    validation_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    hard_mass: torch.Tensor,
    batch_size: int,
) -> tuple[float, list[dict[str, object]]]:
    grid = []
    for alpha in ALPHAS:
        score = blend_score(identity_logits, probe_logit, mask, alpha)
        metrics = evaluate_score(
            score,
            store,
            validation_rows,
            normalization,
            hard_mass,
            batch_size,
            False,
        )
        grid.append({"name": name, "alpha": alpha, "metrics": metrics})
    best = max(grid, key=lambda item: float(item["metrics"]["selection_score"]))
    return float(best["alpha"]), grid


def candidate_ap(
    score: np.ndarray,
    active: np.ndarray,
    mask: np.ndarray,
    rows: np.ndarray,
) -> float:
    valid = mask[rows]
    return float(
        average_precision_score(
            active[rows][valid].astype(np.int8), score[rows][valid]
        )
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    directory = e118.cache_dir(e123.cache_tag(OUTER))
    store = e118.ClusterTensorStore(directory, device)
    identity_model, normalization, _ = e126.load_identity_core(store, device)
    identity_logits, identity_mass = identity_predictions(
        identity_model, store, normalization, args.batch_size
    )
    del identity_model

    component = np.load(e114.COMPONENTS, allow_pickle=False)
    coordinate = np.asarray(
        np.load(directory / "coordinate.npy", mmap_mode="r"), np.float32
    )
    mask = np.asarray(np.load(directory / "mask.npy", mmap_mode="r"), bool)
    base_log_mass = np.asarray(
        np.load(directory / "base_log_mass.npy", mmap_mode="r"), np.float32
    )
    member_count = np.asarray(
        np.load(directory / "member_count.npy", mmap_mode="r"), np.int16
    )
    targets, assignment = e124.build_one_to_one_targets(
        coordinate,
        mask,
        component["coordinates"].astype(np.float32),
        component["coefficients"].astype(np.complex64),
        component["point_power"].astype(np.float64),
    )
    target_absolute = targets["mass_absolute"].astype(np.float32)
    target_total = target_absolute.sum(axis=1, keepdims=True)
    target_mass = np.divide(
        target_absolute,
        np.maximum(target_total, TINY),
        out=np.zeros_like(target_absolute),
    )
    target_active = target_absolute > 0.0
    hard_mass = torch.as_tensor(target_absolute, device=device)

    _, positions, _, site, _ = load_positions()
    fold_id = np.load(e115.FULL_FOLD).astype(np.int8)
    fit_rows = np.flatnonzero((fold_id != OUTER) & (fold_id != INNER))
    validation_rows = np.flatnonzero(fold_id == INNER)
    held_rows = np.flatnonzero(fold_id == OUTER)
    neighbor_ids, shuffled_neighbor_ids, neighbor_distance = build_neighbors(
        positions, site, fold_id, args.seed
    )

    identity_probability = 1.0 / (
        1.0 + np.exp(-np.clip(identity_logits, -30.0, 30.0))
    )
    correct_lifecycle = lifecycle_features(
        coordinate,
        mask,
        identity_mass,
        identity_probability,
        neighbor_ids,
        neighbor_distance,
    )
    shuffled_lifecycle = lifecycle_features(
        coordinate,
        mask,
        identity_mass,
        identity_probability,
        shuffled_neighbor_ids,
        neighbor_distance,
    )
    oracle_correct = lifecycle_features(
        coordinate,
        mask,
        target_mass,
        target_active.astype(np.float32),
        neighbor_ids,
        neighbor_distance,
    )
    oracle_shuffled = lifecycle_features(
        coordinate,
        mask,
        target_mass,
        target_active.astype(np.float32),
        shuffled_neighbor_ids,
        neighbor_distance,
    )

    unary = unary_features(
        identity_logits, identity_mass, base_log_mass, member_count, mask
    )
    correct_features = np.concatenate((unary, correct_lifecycle), axis=2)
    shuffled_features = np.concatenate((unary, shuffled_lifecycle), axis=2)
    unary_probe = fit_probe(unary, target_active, mask, fit_rows)
    correct_probe = fit_probe(correct_features, target_active, mask, fit_rows)
    shuffled_probe = fit_probe(shuffled_features, target_active, mask, fit_rows)
    unary_logit = probe_scores(unary_probe, unary, mask)
    correct_logit = probe_scores(correct_probe, correct_features, mask)
    shuffled_logit = probe_scores(shuffled_probe, shuffled_features, mask)

    selected_alpha = {}
    selection_grids = {}
    for name, probe_logit in (
        ("unary_probe", unary_logit),
        ("correct_lifecycle_probe", correct_logit),
        ("shuffled_lifecycle_probe", shuffled_logit),
    ):
        alpha, grid = select_alpha(
            name,
            probe_logit,
            identity_logits,
            mask,
            store,
            validation_rows,
            normalization,
            hard_mass,
            args.batch_size,
        )
        selected_alpha[name] = alpha
        selection_grids[name] = grid

    held_metrics = {
        "identity_reference": evaluate_score(
            identity_logits,
            store,
            held_rows,
            normalization,
            hard_mass,
            args.batch_size,
            True,
        )
    }
    for name, probe_logit in (
        ("unary_probe", unary_logit),
        ("correct_lifecycle_probe", correct_logit),
        ("shuffled_lifecycle_probe", shuffled_logit),
    ):
        held_metrics[name] = evaluate_score(
            blend_score(
                identity_logits, probe_logit, mask, selected_alpha[name]
            ),
            store,
            held_rows,
            normalization,
            hard_mass,
            args.batch_size,
            True,
        )

    held_probe_ap = {
        "identity_raw": candidate_ap(
            identity_logits, target_active, mask, held_rows
        ),
        "unary_probe_raw": candidate_ap(
            unary_logit, target_active, mask, held_rows
        ),
        "correct_lifecycle_probe_raw": candidate_ap(
            correct_logit, target_active, mask, held_rows
        ),
        "shuffled_lifecycle_probe_raw": candidate_ap(
            shuffled_logit, target_active, mask, held_rows
        ),
        "oracle_neighbor_activity_correct": candidate_ap(
            oracle_correct[..., 3], target_active, mask, held_rows
        ),
        "oracle_neighbor_activity_shuffled": candidate_ap(
            oracle_shuffled[..., 3], target_active, mask, held_rows
        ),
    }

    reference = held_metrics["identity_reference"]
    correct = held_metrics["correct_lifecycle_probe"]
    ap_margin = (
        held_probe_ap["correct_lifecycle_probe_raw"]
        - held_probe_ap["shuffled_lifecycle_probe_raw"]
    )
    top16_gain = (
        correct["top16_target_mass_recall"]
        - reference["top16_target_mass_recall"]
    )
    full_gain = (
        correct["full_official"]["score"]
        - reference["full_official"]["score"]
    )
    gate = {
        "minimum_correct_vs_shuffled_existence_ap_margin": 0.01,
        "require_positive_top16_mass_recall_gain": True,
        "require_positive_identity_full_score_gain": True,
        "correct_vs_shuffled_existence_ap_margin": float(ap_margin),
        "top16_mass_recall_gain_over_identity": float(top16_gain),
        "identity_full_score_gain_over_identity": float(full_gain),
    }
    gate["passed"] = bool(ap_margin >= 0.01 and top16_gain > 0.0 and full_gain > 0.0)

    expected_identity_score = 0.4948253852127809
    replay_error = abs(
        float(reference["full_official"]["score"]) - expected_identity_score
    )
    if replay_error > 2e-6:
        raise RuntimeError(f"E125 identity replay drifted by {replay_error:.3e}")

    np.savez_compressed(
        METRICS,
        fold_id=fold_id,
        neighbor_ids=neighbor_ids,
        shuffled_neighbor_ids=shuffled_neighbor_ids,
        neighbor_distance=neighbor_distance,
        identity_logits=identity_logits.astype(np.float32),
        unary_probe_logit=unary_logit,
        correct_lifecycle_probe_logit=correct_logit,
        shuffled_lifecycle_probe_logit=shuffled_logit,
        target_active=target_active,
        target_mass=target_mass,
    )
    report = {
        "experiment": VERSION,
        "status": "information_audit_complete",
        "primary_hypothesis": (
            "Explicit same-fold neighboring-query path persistence contains candidate "
            "identity information beyond the frozen E125 unary core."
        ),
        "strict_protocol": {
            "outer": OUTER,
            "inner": INNER,
            "fit_count": int(len(fit_rows)),
            "validation_count": int(len(validation_rows)),
            "held_count": int(len(held_rows)),
            "held_used_for_probe_or_alpha_selection": False,
            "candidate_donor_excludes_query_fold": True,
            "official_h_only": True,
            "sionna_or_v12_used": False,
        },
        "fixed_design": {
            "neighbors": NEIGHBORS,
            "batch_size": args.batch_size,
            "spatial_scale_m": SPATIAL_SCALE_M,
            "match_bandwidth": MATCH_BANDWIDTH,
            "alphas_selected_on_inner": list(ALPHAS),
            "probe": "standardized class-balanced logistic regression, C=1",
            "shuffle_control": (
                "eight unique same-site same-fold random rows exclude the query and "
                "its eight correct neighbors while correct-neighbor distance weights "
                "are preserved"
            ),
            "shuffle_seed": args.seed,
            "shuffle_control_integrity": {
                "self_evidence_count": int(
                    np.sum(shuffled_neighbor_ids == np.arange(len(fold_id))[:, None])
                ),
                "correct_neighbor_overlap_count": int(
                    sum(
                        len(set(neighbor_ids[row]) & set(shuffled_neighbor_ids[row]))
                        for row in range(len(fold_id))
                    )
                ),
                "site_match": bool(
                    np.all(site[shuffled_neighbor_ids] == site[:, None])
                ),
                "fold_match": bool(
                    np.all(fold_id[shuffled_neighbor_ids] == fold_id[:, None])
                ),
                "correct_distance_mean_m": float(np.mean(neighbor_distance)),
                "shuffled_distance_mean_m": float(
                    np.mean(
                        np.linalg.norm(
                            positions[shuffled_neighbor_ids, :2]
                            - positions[:, None, :2],
                            axis=2,
                        )
                    )
                ),
            },
        },
        "assignment": assignment,
        "probe_models": {
            "unary": probe_payload(unary_probe),
            "correct_lifecycle": probe_payload(correct_probe),
            "shuffled_lifecycle": probe_payload(shuffled_probe),
        },
        "selected_alpha": selected_alpha,
        "inner_selection_grids": selection_grids,
        "held_probe_ap": held_probe_ap,
        "held_metrics": held_metrics,
        "identity_replay_absolute_error": float(replay_error),
        "gate": gate,
        "decision": {
            "design_candidate_lifecycle_network": bool(gate["passed"]),
            "next_if_passed": (
                "Replace the generic averaged graph branch with explicit candidate "
                "data association and birth/stable/death/null messages."
            ),
            "next_if_failed": (
                "Do not build a larger lifecycle GNN; inspect path-aligned scene "
                "evidence or use Sionna only to identify a different state variable."
            ),
        },
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "metrics_bytes": int(METRICS.stat().st_size),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_alpha": selected_alpha,
                "held_probe_ap": held_probe_ap,
                "held_metrics": held_metrics,
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
