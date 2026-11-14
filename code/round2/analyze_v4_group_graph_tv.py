#!/usr/bin/env python3
"""Evaluate robust groupwise graph diffusion across complete Round 2 holes."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from data import load_positions
from project_paths import PATHS
from run_v3_grouped_link_refinement import build_grouped_descriptors
from run_v4_group_conditioned_link_field import load_state_features


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_group_graph_tv"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_group_graph_tv"
REPORT = ANALYSIS_DIR / "group_graph_tv.json"
POINT_CACHE = ARTIFACT_DIR / "group_graph_tv_oof_scores.npz"

FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
OOF_WEIGHTS = (
    PATHS.artifacts / "round2" / "rbf_spectral_field" / "round2_rbf_oof_weights.npz"
)
DESCRIPTORS = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_spectral_descriptors.npz"
)

BOUNDARY_NEIGHBORS = 128
GRAPH_NEIGHBORS = 12
PCA_COMPONENTS = 12
GEOMETRY_STRENGTH = 0.5
DIFFUSION_STRENGTHS = (0.10, 0.30, 1.00, 3.00, 10.00)
ROBUST_EPSILONS = (0.02, 0.05, 0.10)
BLEND_ALPHAS = (0.25, 0.50, 1.00)
ITERATIONS = 16


@dataclass(frozen=True)
class Config:
    strength: float
    epsilon: float
    alpha: float

    @property
    def name(self) -> str:
        return f"s{self.strength:g}_e{self.epsilon:g}_a{self.alpha:g}"


def normalize_amplitude(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    norm = np.sqrt(np.sum(output * output, axis=-1, keepdims=True, dtype=np.float64))
    return np.divide(
        output,
        np.maximum(norm, 1e-20),
        out=np.zeros_like(output),
        where=norm > 0.0,
    )


def power_cosine(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    predicted_power = np.square(np.maximum(prediction, 0.0))
    truth_power = np.square(np.maximum(truth, 0.0))
    numerator = np.sum(predicted_power * truth_power, axis=-1, dtype=np.float64)
    product = np.sum(predicted_power**2, axis=-1, dtype=np.float64) * np.sum(
        truth_power**2, axis=-1, dtype=np.float64
    )
    return np.divide(
        numerator,
        np.sqrt(np.maximum(product, 1e-30)),
        out=np.zeros_like(numerator),
        where=product > 0.0,
    ).astype(np.float32)


def v2_anchor(
    amplitude: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(donor_ids), amplitude.shape[1], amplitude.shape[2]), dtype=np.float32)
    for row, (ids, values) in enumerate(zip(donor_ids, weights)):
        output[row] = np.maximum(
            np.einsum(
                "k,kgd->gd",
                values.astype(np.float32),
                np.asarray(amplitude[ids], dtype=np.float32),
            ),
            0.0,
        )
    return normalize_amplitude(output)


def boundary_ids(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    fit = fit_ids[valid[fit_ids]]
    take = min(BOUNDARY_NEIGHBORS, len(fit))
    local = cKDTree(positions[fit, :2]).query(
        positions[query_ids, :2], k=take, workers=-1
    )[1]
    return np.unique(fit[np.asarray(local).reshape(-1)])


def graph_edges(
    xy: np.ndarray,
    state: np.ndarray,
    fit_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(state[:fit_count], axis=0, dtype=np.float64)
    std = np.std(state[:fit_count], axis=0, dtype=np.float64)
    standardized = np.clip((state - mean) / np.maximum(std, 1e-4), -8.0, 8.0)
    components = min(PCA_COMPONENTS, standardized.shape[1], max(fit_count - 1, 1))
    embedded = PCA(n_components=components, random_state=20_260_805).fit(
        standardized[:fit_count]
    ).transform(standardized)

    take = min(GRAPH_NEIGHBORS + 1, len(xy))
    distance, neighbor = cKDTree(xy).query(xy, k=take, workers=-1)
    distance = np.asarray(distance[:, 1:], dtype=np.float64)
    neighbor = np.asarray(neighbor[:, 1:], dtype=np.int64)
    source = np.repeat(np.arange(len(xy), dtype=np.int64), distance.shape[1])
    target = neighbor.reshape(-1)
    local_scale = np.maximum(distance[:, min(7, distance.shape[1] - 1)], 1.0)
    denominator = np.maximum(local_scale[source] * local_scale[target], 1e-8)
    weight = np.exp(-np.square(distance.reshape(-1)) / denominator)
    geometry_distance = np.sum(np.square(embedded[source] - embedded[target]), axis=1)
    positive = geometry_distance[geometry_distance > 1e-10]
    geometry_scale = float(np.median(positive)) if len(positive) else 1.0
    weight *= np.exp(
        -GEOMETRY_STRENGTH * geometry_distance / max(geometry_scale, 1e-8)
    )
    reverse_source = np.concatenate([source, target])
    reverse_target = np.concatenate([target, source])
    reverse_weight = np.concatenate([weight, weight]).astype(np.float32)
    packed = reverse_source.astype(np.int64) * len(xy) + reverse_target.astype(np.int64)
    order = np.argsort(packed)
    packed = packed[order]
    reverse_weight = reverse_weight[order]
    unique, first = np.unique(packed, return_index=True)
    maximum = np.maximum.reduceat(reverse_weight, first)
    return (
        (unique // len(xy)).astype(np.int64),
        (unique % len(xy)).astype(np.int64),
        maximum.astype(np.float32),
    )


@torch.no_grad()
def robust_diffusion(
    known: np.ndarray,
    anchor: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    base_weight: np.ndarray,
    config: Config,
    device: torch.device,
) -> np.ndarray:
    known_count = len(known)
    values = torch.as_tensor(
        np.concatenate([known, anchor], axis=0), dtype=torch.float32, device=device
    )
    fixed = values[:known_count].clone()
    anchor_tensor = values[known_count:].clone()
    source_tensor = torch.as_tensor(source, dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.long, device=device)
    base = torch.as_tensor(base_weight, dtype=torch.float32, device=device)
    query_index = torch.arange(known_count, len(values), device=device)
    groups = values.shape[1]
    dimensions = values.shape[2]

    for _ in range(ITERATIONS):
        difference = values[source_tensor] - values[target_tensor]
        rms = torch.sqrt(difference.square().mean(dim=2).clamp_min(1e-12))
        robust = config.epsilon / torch.sqrt(rms.square() + config.epsilon**2)
        edge_weight = base[:, None] * robust
        message = torch.zeros_like(values)
        degree = torch.zeros((len(values), groups), dtype=torch.float32, device=device)
        message.index_add_(
            0,
            source_tensor,
            edge_weight[:, :, None] * values[target_tensor],
        )
        degree.index_add_(0, source_tensor, edge_weight)
        proposal = (
            anchor_tensor
            + config.strength * message[query_index]
        ) / (1.0 + config.strength * degree[query_index, :, None])
        proposal = torch.relu(proposal)
        proposal = proposal / torch.sqrt(
            proposal.square().sum(dim=2, keepdim=True).clamp_min(1e-20)
        )
        values[query_index] = proposal
        values[:known_count] = fixed
    solved = values[query_index]
    blended = (1.0 - config.alpha) * anchor_tensor + config.alpha * solved
    blended = torch.relu(blended)
    blended = blended / torch.sqrt(
        blended.square().sum(dim=2, keepdim=True).clamp_min(1e-20)
    )
    if solved.shape != (len(anchor), groups, dimensions):
        raise RuntimeError(solved.shape)
    return blended.cpu().numpy().astype(np.float32)


def selected_scores(
    score: np.ndarray,
    choice: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    rows = np.flatnonzero(mask)
    groups = score.shape[2]
    return score[
        choice[:, None], rows[None, :], np.arange(groups)[:, None]
    ].T


def summarize_nested(
    score: np.ndarray,
    baseline: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    config_names: list[str],
) -> dict[str, object]:
    nested_global = np.empty_like(baseline)
    nested_group = np.empty_like(baseline)
    global_choices: list[dict[str, object]] = []
    group_choices: list[dict[str, object]] = []
    for held in np.unique(fold_id):
        train = fold_id != held
        held_mask = fold_id == held
        global_index = int(np.argmax(score[:, train].mean(axis=(1, 2))))
        nested_global[held_mask] = score[global_index, held_mask]
        global_choices.append(
            {"held_fold": int(held), "config": config_names[global_index]}
        )
        for site_value in np.unique(site[held_mask]):
            training = train & (site == site_value)
            output = held_mask & (site == site_value)
            choice = np.argmax(score[:, training].mean(axis=1), axis=0)
            nested_group[output] = selected_scores(score, choice, output)
            group_choices.append(
                {
                    "held_fold": int(held),
                    "site": int(site_value),
                    "configs": [config_names[int(value)] for value in choice],
                }
            )

    def result(values: np.ndarray) -> dict[str, object]:
        return {
            "mean": float(np.mean(values)),
            "delta": float(np.mean(values) - np.mean(baseline)),
            "fold_delta": [
                float(
                    np.mean(values[fold_id == fold])
                    - np.mean(baseline[fold_id == fold])
                )
                for fold in np.unique(fold_id)
            ],
            "positive_folds": int(
                sum(
                    np.mean(values[fold_id == fold])
                    > np.mean(baseline[fold_id == fold])
                    for fold in np.unique(fold_id)
                )
            ),
        }

    posterior_index = int(np.argmax(score.mean(axis=(1, 2))))
    return {
        "baseline": float(np.mean(baseline)),
        "posterior_best": {
            "config": config_names[posterior_index],
            **result(score[posterior_index]),
        },
        "nested_global": {**result(nested_global), "choices": global_choices},
        "nested_site_group": {**result(nested_group), "choices": group_choices},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (FOLD_CACHE, OOF_WEIGHTS, DESCRIPTORS):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    _, train_positions, _, train_site, _ = load_positions()
    train_state, _ = load_state_features()
    valid = np.load(DESCRIPTORS)["valid"].astype(bool)
    folds = np.load(FOLD_CACHE)
    weights = np.load(OOF_WEIGHTS)
    query_ids = weights["query_ids"].astype(np.int64)
    fold_id = weights["fold_id"].astype(np.int8)
    donor_ids = weights["neighbor_ids"].astype(np.int64)
    base_weights = weights["weights"].astype(np.float32)
    query_site = train_site[query_ids]
    pas_raw, pdp_raw = build_grouped_descriptors()
    pas_amplitude = normalize_amplitude(np.asarray(pas_raw, dtype=np.float32))
    pdp_amplitude = normalize_amplitude(np.asarray(pdp_raw, dtype=np.float32))
    base_pas = v2_anchor(pas_amplitude, donor_ids, base_weights)
    base_pdp = v2_anchor(pdp_amplitude, donor_ids, base_weights)
    configs = [
        Config(strength, epsilon, alpha)
        for strength in DIFFUSION_STRENGTHS
        for epsilon in ROBUST_EPSILONS
        for alpha in BLEND_ALPHAS
    ]
    pas_prediction = np.empty(
        (len(configs), len(query_ids), pas_amplitude.shape[1], pas_amplitude.shape[2]),
        dtype=np.float32,
    )
    pdp_prediction = np.empty(
        (len(configs), len(query_ids), pdp_amplitude.shape[1], pdp_amplitude.shape[2]),
        dtype=np.float32,
    )

    for held in np.unique(fold_id):
        held_rows = np.flatnonzero(fold_id == held)
        fit_all = folds[f"fold{int(held)}_fit"].astype(np.int64)
        for site_value in np.unique(query_site[held_rows]):
            output_rows = held_rows[query_site[held_rows] == site_value]
            query = query_ids[output_rows]
            fit_site = fit_all[train_site[fit_all] == site_value]
            boundary = boundary_ids(
                fit_site, query, train_positions, valid
            )
            nodes = np.concatenate([boundary, query])
            source, target, graph_weight = graph_edges(
                train_positions[nodes, :2], train_state[nodes], len(boundary)
            )
            known_pas = pas_amplitude[boundary]
            known_pdp = pdp_amplitude[boundary]
            anchor_pas = base_pas[output_rows]
            anchor_pdp = base_pdp[output_rows]
            for config_index, config in enumerate(configs):
                pas_prediction[config_index, output_rows] = robust_diffusion(
                    known_pas,
                    anchor_pas,
                    source,
                    target,
                    graph_weight,
                    config,
                    device,
                )
                pdp_prediction[config_index, output_rows] = robust_diffusion(
                    known_pdp,
                    anchor_pdp,
                    source,
                    target,
                    graph_weight,
                    config,
                    device,
                )
            print(
                f"group-TV fold={int(held)} site={int(site_value)} "
                f"query={len(query)} boundary={len(boundary)} nodes={len(nodes)}",
                flush=True,
            )

    pas_truth = pas_amplitude[query_ids]
    pdp_truth = pdp_amplitude[query_ids]
    base_pas_score = power_cosine(base_pas, pas_truth)
    base_pdp_score = power_cosine(base_pdp, pdp_truth)
    pas_score = np.stack(
        [power_cosine(prediction, pas_truth) for prediction in pas_prediction]
    )
    pdp_score = np.stack(
        [power_cosine(prediction, pdp_truth) for prediction in pdp_prediction]
    )
    config_names = [config.name for config in configs]
    np.savez_compressed(
        POINT_CACHE,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        site=query_site.astype(np.int8),
        config_names=np.asarray(config_names),
        base_pas=base_pas_score,
        base_pdp=base_pdp_score,
        pas=pas_score,
        pdp=pdp_score,
    )
    report = {
        "experiment": "Round2-E14-group-graph-TV",
        "status": "analysis_only",
        "primary_hypothesis": (
            "Robust groupwise graph diffusion can use query-query continuity while "
            "preserving path birth/death discontinuities that linear harmonic extension blurs."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "test_test_used": True,
        "parameters": {
            "boundary_neighbors": BOUNDARY_NEIGHBORS,
            "graph_neighbors": GRAPH_NEIGHBORS,
            "pca_components": PCA_COMPONENTS,
            "geometry_strength": GEOMETRY_STRENGTH,
            "iterations": ITERATIONS,
            "configs": [asdict(config) for config in configs],
        },
        "pas": summarize_nested(
            pas_score, base_pas_score, fold_id, query_site, config_names
        ),
        "pdp": summarize_nested(
            pdp_score, base_pdp_score, fold_id, query_site, config_names
        ),
        "outputs": {
            "point_cache": PATHS.relative(POINT_CACHE),
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
