#!/usr/bin/env python3
"""Transfer fit-only boundary donor policies into Round 2 validation holes."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from data import load_positions
from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_boundary_policy_transfer"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_boundary_policy_transfer"
REPORT = ANALYSIS_DIR / "boundary_policy_transfer.json"
SCORE_CACHE = ARTIFACT_DIR / "boundary_policy_transfer_scores.npz"

V3_DIR = PATHS.artifacts / "round2" / "v3_anchored_neural_rbf"
PAS_AMPLITUDE = V3_DIR / "grouped_pas_amplitude_f16.npy"
PDP_AMPLITUDE = V3_DIR / "grouped_pdp_amplitude_f16.npy"
OOF_WEIGHTS = (
    PATHS.artifacts / "round2" / "rbf_spectral_field" / "round2_rbf_oof_weights.npz"
)
FOLD_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
DESCRIPTORS = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_spectral_descriptors.npz"
)


@dataclass(frozen=True)
class Config:
    donors: int = 64
    source_pool_per_query: int = 32
    source_guard_m: float = 6.0
    source_candidate_neighbors: int = 256
    oracle_steps: int = 80
    oracle_learning_rate: float = 0.12
    oracle_batch_size: int = 8
    transfer_neighbors: tuple[int, ...] = (2, 4, 8, 16, 32)
    blend_alphas: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    seed: int = 20_260_805


def power_cosine(predicted_amplitude: torch.Tensor, truth_amplitude: torch.Tensor) -> torch.Tensor:
    predicted = predicted_amplitude.square()
    truth = truth_amplitude.square()
    numerator = torch.sum(predicted * truth, dim=-1)
    product = (
        torch.sum(predicted.square(), dim=-1)
        * torch.sum(truth.square(), dim=-1)
    )
    return numerator / torch.sqrt(product.clamp_min(1e-20))


def normalized_positive(weights: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(weights, dtype=np.float32), 0.0) + 1e-4
    output /= output.sum(axis=-1, keepdims=True)
    return output


def source_contexts(
    source_ids: np.ndarray,
    fit_ids: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    site: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    donors = np.empty((len(source_ids), config.donors), dtype=np.int32)
    distances = np.empty((len(source_ids), config.donors), dtype=np.float32)
    for site_value in np.unique(site[source_ids]):
        rows = np.flatnonzero(site[source_ids] == site_value)
        bank = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
        count = min(config.source_candidate_neighbors, len(bank))
        candidate_distance, candidate_local = cKDTree(positions[bank, :2]).query(
            positions[source_ids[rows], :2], k=count, workers=-1
        )
        candidate_ids = bank[candidate_local]
        for local_row, output_row in enumerate(rows):
            keep = (
                (candidate_ids[local_row] != source_ids[output_row])
                & (candidate_distance[local_row] >= config.source_guard_m)
            )
            selected_ids = candidate_ids[local_row, keep][: config.donors]
            selected_distance = candidate_distance[local_row, keep][: config.donors]
            if len(selected_ids) != config.donors:
                raise RuntimeError(
                    f"source {source_ids[output_row]} has only {len(selected_ids)} donors"
                )
            donors[output_row] = selected_ids
            distances[output_row] = selected_distance
    return donors, distances


def optimize_source_policies(
    amplitude: np.ndarray,
    source_ids: np.ndarray,
    donor_ids: np.ndarray,
    donor_distances: np.ndarray,
    config: Config,
    device: torch.device,
) -> np.ndarray:
    groups = amplitude.shape[1]
    output = np.empty((len(source_ids), groups, config.donors), dtype=np.float32)
    initial = 1.0 / np.maximum(donor_distances, 1e-3)
    initial /= initial.sum(axis=1, keepdims=True)
    for start in range(0, len(source_ids), config.oracle_batch_size):
        end = min(start + config.oracle_batch_size, len(source_ids))
        donor = torch.as_tensor(
            np.asarray(amplitude[donor_ids[start:end]], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        truth = torch.as_tensor(
            np.asarray(amplitude[source_ids[start:end]], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        start_weights = np.broadcast_to(
            initial[start:end, None],
            (end - start, groups, config.donors),
        ).copy()
        logits = torch.nn.Parameter(
            torch.log(torch.as_tensor(start_weights, device=device).clamp_min(1e-8))
        )
        optimizer = torch.optim.Adam([logits], lr=config.oracle_learning_rate)
        for _ in range(config.oracle_steps):
            weights = torch.softmax(logits, dim=-1)
            prediction = torch.einsum("bgk,bkgd->bgd", weights, donor).clamp_min(0.0)
            loss = 1.0 - power_cosine(prediction, truth).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        output[start:end] = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return output


def transfer_policy(
    query_rows: np.ndarray,
    query_ids: np.ndarray,
    query_donors: np.ndarray,
    source_ids: np.ndarray,
    source_donors: np.ndarray,
    source_policy: np.ndarray,
    positions: np.ndarray,
    base_weights: np.ndarray,
    neighbors: int,
    overlap_weighted: bool,
) -> tuple[np.ndarray, float]:
    source_count = min(neighbors, len(source_ids))
    distance, local = cKDTree(positions[source_ids, :2]).query(
        positions[query_ids[query_rows], :2], k=source_count, workers=-1
    )
    distance = np.asarray(distance)
    local = np.asarray(local)
    if source_count == 1:
        distance = distance[:, None]
        local = local[:, None]
    output = np.empty(
        (len(query_rows), source_policy.shape[1], query_donors.shape[1]),
        dtype=np.float32,
    )
    overlap_values: list[float] = []
    for local_row, global_row in enumerate(query_rows):
        current_ids = query_donors[global_row]
        current_lookup = {int(point): rank for rank, point in enumerate(current_ids)}
        source_weight = 1.0 / np.maximum(distance[local_row], 0.5)
        mapped_rows: list[np.ndarray] = []
        overlap_rows: list[float] = []
        for source_local in local[local_row]:
            mapped = np.zeros((source_policy.shape[1], len(current_ids)), dtype=np.float32)
            hits = 0
            for source_rank, donor_id in enumerate(source_donors[source_local]):
                target_rank = current_lookup.get(int(donor_id))
                if target_rank is not None:
                    mapped[:, target_rank] = source_policy[source_local, :, source_rank]
                    hits += 1
            mapped_rows.append(mapped)
            overlap_rows.append(hits / len(current_ids))
        overlap = np.asarray(overlap_rows, dtype=np.float32)
        if overlap_weighted:
            source_weight *= np.maximum(overlap, 0.05) ** 2
        source_weight /= np.sum(source_weight)
        mapped = np.sum(
            np.asarray(mapped_rows) * source_weight[:, None, None], axis=0
        )
        missing = np.maximum(1.0 - mapped.sum(axis=1, keepdims=True), 0.0)
        anchor = normalized_positive(base_weights[global_row])[None]
        mapped += missing * anchor
        mapped += 1e-8
        mapped /= mapped.sum(axis=1, keepdims=True)
        output[local_row] = mapped
        overlap_values.append(float(np.sum(source_weight * overlap)))
    return output, float(np.mean(overlap_values))


def score_policy(
    policy: np.ndarray,
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    amplitude: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.empty((len(query_ids), amplitude.shape[1]), dtype=np.float32)
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        donor = torch.as_tensor(
            np.asarray(amplitude[donor_ids[start:end]], dtype=np.float32), device=device
        )
        truth = torch.as_tensor(
            np.asarray(amplitude[query_ids[start:end]], dtype=np.float32), device=device
        )
        weights = torch.as_tensor(policy[start:end], device=device)
        prediction = torch.einsum("bgk,bkgd->bgd", weights, donor).clamp_min(0.0)
        output[start:end] = power_cosine(prediction, truth).cpu().numpy()
    return output


def blend_policy(base: np.ndarray, transferred: np.ndarray, alpha: float) -> np.ndarray:
    anchor = normalized_positive(base)[:, None, :]
    output = (1.0 - alpha) * anchor + alpha * transferred
    output /= output.sum(axis=2, keepdims=True)
    return output.astype(np.float32)


def nested_selection(
    scores: np.ndarray,
    names: np.ndarray,
    fold_id: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    selected = np.empty(len(fold_id), dtype=np.float32)
    decisions: list[dict[str, object]] = []
    for held_fold in np.unique(fold_id):
        train = fold_id != held_fold
        held = fold_id == held_fold
        means = scores[:, train].mean(axis=1)
        choice = int(np.argmax(means))
        selected[held] = scores[choice, held]
        decisions.append(
            {
                "held_fold": int(held_fold),
                "config": str(names[choice]),
                "selection_score": float(means[choice]),
                "held_score": float(np.mean(scores[choice, held])),
            }
        )
    return selected, decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--oracle-steps", type=int, default=Config.oracle_steps)
    parser.add_argument("--source-guard-m", type=float, default=Config.source_guard_m)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(oracle_steps=args.oracle_steps, source_guard_m=args.source_guard_m)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (PAS_AMPLITUDE, PDP_AMPLITUDE, OOF_WEIGHTS, FOLD_CACHE, DESCRIPTORS):
        if not path.is_file():
            raise FileNotFoundError(path)
    _, positions, _, site, _ = load_positions()
    valid = np.load(DESCRIPTORS)["valid"].astype(bool)
    oof = np.load(OOF_WEIGHTS)
    query_ids = oof["query_ids"].astype(np.int64)
    fold_id = oof["fold_id"].astype(np.int8)
    query_donors = oof["neighbor_ids"].astype(np.int64)
    base_weights = oof["weights"].astype(np.float32)
    folds = np.load(FOLD_CACHE)
    pas_amplitude = np.load(PAS_AMPLITUDE, mmap_mode="r")
    pdp_amplitude = np.load(PDP_AMPLITUDE, mmap_mode="r")
    device = torch.device(args.device)

    names = ["positive_rbf_anchor"]
    configs = [(0, 0.0, False)]
    for overlap_weighted in (False, True):
        scheme = "overlap" if overlap_weighted else "distance"
        for neighbors in config.transfer_neighbors:
            for alpha in config.blend_alphas:
                names.append(f"{scheme}_k{neighbors}_a{alpha:g}")
                configs.append((neighbors, alpha, overlap_weighted))
    pas_scores = np.empty((len(names), len(query_ids)), dtype=np.float32)
    pdp_scores = np.empty_like(pas_scores)
    pas_policy_by_config = np.empty(
        (len(names), len(query_ids), pas_amplitude.shape[1], config.donors),
        dtype=np.float16,
    )
    pdp_policy_by_config = np.empty(
        (len(names), len(query_ids), pdp_amplitude.shape[1], config.donors),
        dtype=np.float16,
    )
    positive = normalized_positive(base_weights)
    pas_policy_by_config[0] = np.broadcast_to(
        positive[:, None], pas_policy_by_config[0].shape
    ).astype(np.float16)
    pdp_policy_by_config[0] = np.broadcast_to(
        positive[:, None], pdp_policy_by_config[0].shape
    ).astype(np.float16)
    overlap_log: list[dict[str, object]] = []

    for fold in np.unique(fold_id):
        fit_ids = folds[f"fold{int(fold)}_fit"].astype(np.int64)
        for site_value in np.unique(site):
            query_rows = np.flatnonzero(
                (fold_id == fold) & (site[query_ids] == site_value)
            )
            fit_site = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
            _, source_local = cKDTree(positions[fit_site, :2]).query(
                positions[query_ids[query_rows], :2],
                k=min(config.source_pool_per_query, len(fit_site)),
                workers=-1,
            )
            source_ids = np.unique(fit_site[np.asarray(source_local).reshape(-1)])
            source_donors, source_distances = source_contexts(
                source_ids, fit_ids, positions, valid, site, config
            )
            pas_source_policy = optimize_source_policies(
                pas_amplitude,
                source_ids,
                source_donors,
                source_distances,
                config,
                device,
            )
            pdp_source_policy = optimize_source_policies(
                pdp_amplitude,
                source_ids,
                source_donors,
                source_distances,
                config,
                device,
            )
            for config_index, (neighbors, alpha, overlap_weighted) in enumerate(configs[1:], 1):
                pas_transfer, pas_overlap = transfer_policy(
                    query_rows,
                    query_ids,
                    query_donors,
                    source_ids,
                    source_donors,
                    pas_source_policy,
                    positions,
                    base_weights,
                    neighbors,
                    overlap_weighted,
                )
                pdp_transfer, pdp_overlap = transfer_policy(
                    query_rows,
                    query_ids,
                    query_donors,
                    source_ids,
                    source_donors,
                    pdp_source_policy,
                    positions,
                    base_weights,
                    neighbors,
                    overlap_weighted,
                )
                pas_policy_by_config[config_index, query_rows] = blend_policy(
                    base_weights[query_rows], pas_transfer, alpha
                ).astype(np.float16)
                pdp_policy_by_config[config_index, query_rows] = blend_policy(
                    base_weights[query_rows], pdp_transfer, alpha
                ).astype(np.float16)
                if alpha == 1.0:
                    overlap_log.append(
                        {
                            "fold": int(fold),
                            "site": int(site_value),
                            "scheme": "overlap" if overlap_weighted else "distance",
                            "neighbors": int(neighbors),
                            "pas_overlap": pas_overlap,
                            "pdp_overlap": pdp_overlap,
                            "source_count": int(len(source_ids)),
                        }
                    )
            print(
                f"boundary policies fold={int(fold)} site={int(site_value)} "
                f"queries={len(query_rows)} sources={len(source_ids)}",
                flush=True,
            )

    for config_index, name in enumerate(names):
        pas_scores[config_index] = score_policy(
            np.asarray(pas_policy_by_config[config_index], dtype=np.float32),
            query_ids,
            query_donors,
            pas_amplitude,
            device,
            config.oracle_batch_size,
        ).mean(axis=1)
        pdp_scores[config_index] = score_policy(
            np.asarray(pdp_policy_by_config[config_index], dtype=np.float32),
            query_ids,
            query_donors,
            pdp_amplitude,
            device,
            config.oracle_batch_size,
        ).mean(axis=1)
        print(
            f"score {name}: PAS={pas_scores[config_index].mean():.6f} "
            f"PDP={pdp_scores[config_index].mean():.6f}",
            flush=True,
        )

    name_array = np.asarray(names)
    nested_pas, pas_decisions = nested_selection(pas_scores, name_array, fold_id)
    nested_pdp, pdp_decisions = nested_selection(pdp_scores, name_array, fold_id)
    posterior_pas = int(np.argmax(pas_scores.mean(axis=1)))
    posterior_pdp = int(np.argmax(pdp_scores.mean(axis=1)))
    np.savez_compressed(
        SCORE_CACHE,
        names=name_array,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        pas_scores=pas_scores,
        pdp_scores=pdp_scores,
        pas_policies=pas_policy_by_config,
        pdp_policies=pdp_policy_by_config,
    )
    payload = {
        "experiment": "Round2-E18-boundary-policy-transfer",
        "status": "analysis_only",
        "primary_hypothesis": (
            "Fit-only boundary points reveal a local donor-selection policy whose shared donor "
            "identities can be transferred into a spatial hole without observing query labels."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "baseline": {
            "pas": float(np.mean(pas_scores[0])),
            "pdp": float(np.mean(pdp_scores[0])),
        },
        "posterior_best": {
            "pas": {
                "config": names[posterior_pas],
                "score": float(np.mean(pas_scores[posterior_pas])),
                "delta": float(np.mean(pas_scores[posterior_pas] - pas_scores[0])),
            },
            "pdp": {
                "config": names[posterior_pdp],
                "score": float(np.mean(pdp_scores[posterior_pdp])),
                "delta": float(np.mean(pdp_scores[posterior_pdp] - pdp_scores[0])),
            },
        },
        "nested": {
            "pas": {
                "score": float(np.mean(nested_pas)),
                "delta": float(np.mean(nested_pas - pas_scores[0])),
                "fold_delta": [
                    float(np.mean((nested_pas - pas_scores[0])[fold_id == fold]))
                    for fold in np.unique(fold_id)
                ],
                "decisions": pas_decisions,
            },
            "pdp": {
                "score": float(np.mean(nested_pdp)),
                "delta": float(np.mean(nested_pdp - pdp_scores[0])),
                "fold_delta": [
                    float(np.mean((nested_pdp - pdp_scores[0])[fold_id == fold]))
                    for fold in np.unique(fold_id)
                ],
                "decisions": pdp_decisions,
            },
        },
        "overlap_diagnostics": overlap_log,
        "outputs": {
            "scores": PATHS.relative(SCORE_CACHE),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
