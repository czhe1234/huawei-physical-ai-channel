#!/usr/bin/env python3
"""Audit whether E12 groupwise donor-oracle weights form a predictable field."""

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


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_group_weight_predictability"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_group_weight_predictability"
REPORT = ANALYSIS_DIR / "group_weight_predictability.json"
WEIGHT_CACHE = ARTIFACT_DIR / "oof_group_oracle_weights.npz"

V3_DIR = PATHS.artifacts / "round2" / "v3_anchored_neural_rbf"
PAS_AMPLITUDE = V3_DIR / "grouped_pas_amplitude_f16.npy"
PDP_AMPLITUDE = V3_DIR / "grouped_pdp_amplitude_f16.npy"
OOF_WEIGHTS = (
    PATHS.artifacts / "round2" / "rbf_spectral_field" / "round2_rbf_oof_weights.npz"
)


@dataclass(frozen=True)
class Config:
    steps: int = 120
    learning_rate: float = 0.12
    batch_size: int = 8
    low_ranks: tuple[int, ...] = (0, 1, 2, 4, 8)
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


def predict_amplitude(weights: torch.Tensor, donors: torch.Tensor) -> torch.Tensor:
    predicted = torch.einsum("bgk,bgkl->bgl", weights, donors).clamp_min(0.0)
    return predicted / torch.sqrt(predicted.square().sum(dim=-1, keepdim=True).clamp_min(1e-20))


def initial_simplex(base_weights: np.ndarray, groups: int) -> np.ndarray:
    positive = np.maximum(np.asarray(base_weights, dtype=np.float32), 0.0) + 1e-4
    positive /= positive.sum(axis=1, keepdims=True)
    return np.broadcast_to(positive[:, None, :], (len(positive), groups, positive.shape[1])).copy()


def optimize_simplex(
    donor_amplitude: np.ndarray,
    truth_amplitude: np.ndarray,
    initial: np.ndarray,
    config: Config,
    device: torch.device,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    count, groups, donors, _ = donor_amplitude.shape
    output = np.empty((count, groups, donors), dtype=np.float32)
    scores = np.empty((count, groups), dtype=np.float32)
    mask_tensor = None
    if fit_mask is not None:
        mask_tensor = torch.as_tensor(fit_mask, dtype=torch.bool, device=device)
    for start in range(0, count, config.batch_size):
        end = min(start + config.batch_size, count)
        donor = torch.as_tensor(donor_amplitude[start:end], dtype=torch.float32, device=device)
        truth = torch.as_tensor(truth_amplitude[start:end], dtype=torch.float32, device=device)
        logits = torch.nn.Parameter(
            torch.log(
                torch.as_tensor(initial[start:end], dtype=torch.float32, device=device).clamp_min(1e-8)
            )
        )
        optimizer = torch.optim.Adam([logits], lr=config.learning_rate)
        local_mask = mask_tensor
        for _ in range(config.steps):
            weights = torch.softmax(logits, dim=-1)
            prediction = predict_amplitude(weights, donor)
            if local_mask is None:
                similarity = power_cosine(prediction, truth)
            else:
                similarity = power_cosine(prediction[..., local_mask], truth[..., local_mask])
            loss = 1.0 - similarity.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            weights = torch.softmax(logits, dim=-1)
            prediction = predict_amplitude(weights, donor)
            output[start:end] = weights.cpu().numpy()
            scores[start:end] = power_cosine(prediction, truth).cpu().numpy()
        print(f"group oracle {end}/{count}", flush=True)
    return output, scores


def score_weights(
    weights: np.ndarray,
    donor_amplitude: np.ndarray,
    truth_amplitude: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(weights.shape[:2], dtype=np.float32)
    for start in range(0, len(weights), batch_size):
        end = min(start + batch_size, len(weights))
        with torch.inference_mode():
            value = predict_amplitude(
                torch.as_tensor(weights[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(donor_amplitude[start:end], dtype=torch.float32, device=device),
            )
            scores[start:end] = power_cosine(
                value,
                torch.as_tensor(truth_amplitude[start:end], dtype=torch.float32, device=device),
            ).cpu().numpy()
    return scores


def low_rank_weights(weights: np.ndarray, rank: int) -> np.ndarray:
    output = np.empty_like(weights)
    for row, matrix in enumerate(weights):
        mean = matrix.mean(axis=0, keepdims=True)
        if rank == 0:
            reconstructed = np.broadcast_to(mean, matrix.shape).copy()
        else:
            centered = matrix - mean
            u, singular, vh = np.linalg.svd(centered, full_matrices=False)
            use = min(rank, len(singular))
            reconstructed = mean + (u[:, :use] * singular[:use]) @ vh[:use]
        reconstructed = np.maximum(reconstructed, 0.0)
        reconstructed += 1e-8
        reconstructed /= reconstructed.sum(axis=1, keepdims=True)
        output[row] = reconstructed
    return output


def grouped_shared_weights(weights: np.ndarray, metric: str) -> dict[str, np.ndarray]:
    if metric == "pas":
        shaped = weights.reshape(len(weights), 4, 8, weights.shape[-1])
        return {
            "point_shared": np.broadcast_to(
                shaped.mean(axis=(1, 2), keepdims=True), shaped.shape
            ).reshape(weights.shape).copy(),
            "ue_shared_frequency": np.broadcast_to(
                shaped.mean(axis=2, keepdims=True), shaped.shape
            ).reshape(weights.shape).copy(),
            "frequency_shared_ue": np.broadcast_to(
                shaped.mean(axis=1, keepdims=True), shaped.shape
            ).reshape(weights.shape).copy(),
        }
    shaped = weights.reshape(len(weights), 2, 4, weights.shape[-1])
    return {
        "point_shared": np.broadcast_to(
            shaped.mean(axis=(1, 2), keepdims=True), shaped.shape
        ).reshape(weights.shape).copy(),
        "polarization_shared_ue": np.broadcast_to(
            shaped.mean(axis=2, keepdims=True), shaped.shape
        ).reshape(weights.shape).copy(),
        "ue_shared_polarization": np.broadcast_to(
            shaped.mean(axis=1, keepdims=True), shaped.shape
        ).reshape(weights.shape).copy(),
    }


def shuffled_weights(
    weights: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty_like(weights)
    for fold in np.unique(fold_id):
        for site_value in np.unique(site):
            rows = np.flatnonzero((fold_id == fold) & (site == site_value))
            output[rows] = weights[rng.permutation(rows)]
    return output


def weight_moments(
    weights: np.ndarray,
    query_xy: np.ndarray,
    donor_xy: np.ndarray,
) -> np.ndarray:
    delta = donor_xy - query_xy[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    rank = np.linspace(0.0, 1.0, weights.shape[2], dtype=np.float32)
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-20)), axis=2)
    entropy /= np.log(weights.shape[2])
    return np.stack(
        [
            np.sum(weights * delta[:, None, :, 0], axis=2),
            np.sum(weights * delta[:, None, :, 1], axis=2),
            np.sum(weights * distance[:, None, :], axis=2),
            np.sum(weights * rank[None, None, :], axis=2),
            entropy,
        ],
        axis=2,
    ).astype(np.float32)


def spatial_moment_control(
    moments: np.ndarray,
    positions: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    nearest_errors: list[np.ndarray] = []
    shuffled_errors: list[np.ndarray] = []
    for fold in np.unique(fold_id):
        for site_value in np.unique(site):
            rows = np.flatnonzero((fold_id == fold) & (site == site_value))
            if len(rows) < 3:
                continue
            nearest = cKDTree(positions[rows, :2]).query(
                positions[rows, :2], k=2, workers=-1
            )[1][:, 1]
            scale = np.std(moments[rows], axis=0, keepdims=True) + 1e-4
            nearest_errors.append(
                np.mean(np.abs((moments[rows] - moments[rows[nearest]]) / scale), axis=(1, 2))
            )
            shuffled = rng.permutation(len(rows))
            shuffled_errors.append(
                np.mean(np.abs((moments[rows] - moments[rows[shuffled]]) / scale), axis=(1, 2))
            )
    near = np.concatenate(nearest_errors)
    control = np.concatenate(shuffled_errors)
    return {
        "nearest_standardized_l1": float(np.mean(near)),
        "within_region_shuffled_standardized_l1": float(np.mean(control)),
        "relative_reduction": float(1.0 - np.mean(near) / max(float(np.mean(control)), 1e-20)),
    }


def summarize_scores(values: np.ndarray, fold_id: np.ndarray) -> dict[str, object]:
    point = values.mean(axis=1)
    return {
        "mean": float(np.mean(point)),
        "fold": [float(np.mean(point[fold_id == fold])) for fold in np.unique(fold_id)],
    }


def metric_analysis(
    metric: str,
    amplitude_cache: np.ndarray,
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    base_weights: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    train_positions: np.ndarray,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[dict[str, object], np.ndarray]:
    truth = np.asarray(amplitude_cache[query_ids], dtype=np.float32)
    donor = np.asarray(amplitude_cache[donor_ids], dtype=np.float32).transpose(0, 2, 1, 3)
    initial = initial_simplex(base_weights, truth.shape[1])
    base_score = score_weights(initial, donor, truth, device, config.batch_size)
    oracle, oracle_score = optimize_simplex(donor, truth, initial, config, device)

    rows: dict[str, object] = {
        "positive_rbf_anchor": summarize_scores(base_score, fold_id),
        "continuous_simplex_oracle": summarize_scores(oracle_score, fold_id),
    }
    for name, shared in grouped_shared_weights(oracle, metric).items():
        rows[name] = summarize_scores(
            score_weights(shared, donor, truth, device, config.batch_size), fold_id
        )
    for rank in config.low_ranks:
        reconstructed = low_rank_weights(oracle, rank)
        rows[f"low_rank_{rank}"] = summarize_scores(
            score_weights(reconstructed, donor, truth, device, config.batch_size), fold_id
        )
    shuffled = shuffled_weights(oracle, fold_id, site, rng)
    rows["within_region_shuffled"] = summarize_scores(
        score_weights(shuffled, donor, truth, device, config.batch_size), fold_id
    )

    moments = weight_moments(
        oracle,
        train_positions[query_ids, :2],
        train_positions[donor_ids, :2],
    )
    report = {
        "scores": rows,
        "oracle_effective_donors": float(
            np.mean(np.exp(-np.sum(oracle * np.log(np.maximum(oracle, 1e-20)), axis=2)))
        ),
        "oracle_peak_weight": float(np.mean(np.max(oracle, axis=2))),
        "spatial_moment_control": spatial_moment_control(
            moments,
            train_positions[query_ids],
            fold_id,
            site,
            rng,
        ),
    }
    return report, oracle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(steps=args.steps, batch_size=args.batch_size)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (PAS_AMPLITUDE, PDP_AMPLITUDE, OOF_WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    _, train_positions, _, train_site, _ = load_positions()
    cache = np.load(OOF_WEIGHTS)
    query_ids = cache["query_ids"].astype(np.int64)
    fold_id = cache["fold_id"].astype(np.int8)
    donor_ids = cache["neighbor_ids"].astype(np.int64)
    base_weights = cache["weights"].astype(np.float32)
    site = train_site[query_ids]
    device = torch.device(args.device)
    rng = np.random.default_rng(config.seed)

    pas_report, pas_weights = metric_analysis(
        "pas",
        np.load(PAS_AMPLITUDE, mmap_mode="r"),
        query_ids,
        donor_ids,
        base_weights,
        fold_id,
        site,
        train_positions,
        config,
        device,
        rng,
    )
    pdp_report, pdp_weights = metric_analysis(
        "pdp",
        np.load(PDP_AMPLITUDE, mmap_mode="r"),
        query_ids,
        donor_ids,
        base_weights,
        fold_id,
        site,
        train_positions,
        config,
        device,
        rng,
    )
    np.savez_compressed(
        WEIGHT_CACHE,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        site=site,
        donor_ids=donor_ids.astype(np.int32),
        pas_weights=pas_weights.astype(np.float16),
        pdp_weights=pdp_weights.astype(np.float16),
    )
    payload = {
        "experiment": "Round2-E17-group-weight-predictability",
        "status": "analysis_only",
        "primary_hypothesis": (
            "The E12 linkwise ceiling is deployable only if continuous group donor weights "
            "have low-rank and spatially stable structure beyond a truth-conditioned selection effect."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "pas": pas_report,
        "pdp": pdp_report,
        "outputs": {
            "weights": PATHS.relative(WEIGHT_CACHE),
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
