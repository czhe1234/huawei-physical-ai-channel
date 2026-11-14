#!/usr/bin/env python3
"""Train the grouped residual field on translated target-shaped analog holes."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from data import load_positions
from project_paths import PATHS
from run_v3_anchored_neural_rbf import FOLD_CACHE, OOF_META, set_seed
from run_v3_grouped_link_refinement import GROUPED_PAS, GROUPED_PDP
from run_v4_group_conditioned_link_field import load_state_features, standardize_state
from run_v4_masked_hole_neural_process import positional_features
from run_v7_grouped_anchor_hole_field import (
    GroupedAnchorHoleField,
    GroupedPCA,
    Config as ModelConfig,
    fit_grouped_pca,
    grouped_cosine,
    normalize_groups,
)


VERSION = "Round2-E54-analog-hole-grouped-field"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "analog_hole_grouped_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_analog_hole_grouped_field"
REPORT = ANALYSIS_DIR / "analog_hole_grouped_field.json"
CONTEXT_DIR = PATHS.artifacts / "round2" / "contiguous_hole_projection_policy"
V3_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "v3_anchored_neural_rbf"
    / "oof_epoch6_corrected_weights.npz"
)
DESCRIPTORS = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_spectral_descriptors.npz"
)
MAX_CONTEXT = 384


@dataclass(frozen=True)
class Config:
    steps: int = 300
    validation_interval: int = 25
    learning_rate: float = 1.5e-4
    weight_decay: float = 3e-5
    action_regularization: float = 0.004
    seed: int = 20_260_806


@dataclass(frozen=True)
class HoleEpisode:
    query_ids: np.ndarray
    donor_ids: np.ndarray
    pas_weights: np.ndarray
    pdp_weights: np.ndarray
    context_ids: np.ndarray


def context_ids(donor_ids: np.ndarray, query_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    candidates = np.unique(donor_ids[:, :16])
    candidates = candidates[~np.isin(candidates, query_ids)]
    if len(candidates) <= MAX_CONTEXT:
        return candidates.astype(np.int64)
    from scipy.spatial import cKDTree

    distance = cKDTree(positions[query_ids, :2]).query(
        positions[candidates, :2], k=1, workers=-1
    )[0]
    return candidates[np.argsort(distance)[:MAX_CONTEXT]].astype(np.int64)


def load_analog_episodes(fold: int, positions: np.ndarray) -> list[HoleEpisode]:
    path = CONTEXT_DIR / f"fold{fold}_contiguous_context.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    cached = np.load(path)
    output: list[HoleEpisode] = []
    for hole_id in np.unique(cached["hole_id"]):
        rows = np.flatnonzero(cached["hole_id"] == hole_id)
        query_ids = cached["query_ids"][rows].astype(np.int64)
        donor_ids = cached["donor_ids"][rows].astype(np.int64)
        output.append(
            HoleEpisode(
                query_ids=query_ids,
                donor_ids=donor_ids,
                pas_weights=cached["pas_weights"][rows].astype(np.float32),
                pdp_weights=cached["pdp_weights"][rows].astype(np.float32),
                context_ids=context_ids(donor_ids, query_ids, positions),
            )
        )
    return output


def load_held_episodes(fold: int, positions: np.ndarray) -> list[HoleEpisode]:
    meta = np.load(OOF_META)
    query_ids_all = meta["query_ids"].astype(np.int64)
    rows_all = np.flatnonzero(meta["fold_id"] == fold)
    weights = np.load(V3_WEIGHTS)
    _, _, _, site, _ = load_positions()
    output: list[HoleEpisode] = []
    for site_value in np.unique(site[query_ids_all[rows_all]]):
        rows = rows_all[site[query_ids_all[rows_all]] == site_value]
        query_ids = query_ids_all[rows]
        donor_ids = weights["donor_ids"][rows].astype(np.int64)
        output.append(
            HoleEpisode(
                query_ids=query_ids,
                donor_ids=donor_ids,
                pas_weights=weights["pas_weights"][rows].astype(np.float32),
                pdp_weights=weights["pdp_weights"][rows].astype(np.float32),
                context_ids=context_ids(donor_ids, query_ids, positions),
            )
        )
    return output


def anchor_grouped(
    value: torch.Tensor,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    ids = torch.as_tensor(donor_ids, dtype=torch.long, device=device)
    row_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return normalize_groups(torch.einsum("qk,qkgd->qgd", row_weights, value[ids]))


def run_episode(
    model: GroupedAnchorHoleField,
    episode: HoleEpisode,
    state: torch.Tensor,
    position: torch.Tensor,
    pas: torch.Tensor,
    pdp: torch.Tensor,
    pca: GroupedPCA,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_ids = torch.as_tensor(episode.query_ids, dtype=torch.long, device=device)
    context = torch.as_tensor(episode.context_ids, dtype=torch.long, device=device)
    anchor_pas = anchor_grouped(pas, episode.donor_ids, episode.pas_weights, device)
    anchor_pdp = anchor_grouped(pdp, episode.donor_ids, episode.pdp_weights, device)
    anchor_flat = torch.cat([anchor_pas.flatten(1), anchor_pdp.flatten(1)], dim=1)
    anchor_coefficients = (anchor_flat - pca.mean) @ pca.vectors
    pas_gain, pdp_gain = model(
        state[query_ids],
        position[query_ids, :2],
        anchor_coefficients,
        state[context],
        position[context, :2],
        pca.coefficients[context],
    )
    prediction_pas = normalize_groups(anchor_pas * torch.exp(pas_gain))
    prediction_pdp = normalize_groups(anchor_pdp * torch.exp(pdp_gain))
    return prediction_pas, prediction_pdp, pas_gain, pdp_gain, query_ids, anchor_pas, anchor_pdp


@torch.no_grad()
def evaluate(
    model: GroupedAnchorHoleField,
    episodes: list[HoleEpisode],
    state: torch.Tensor,
    position: torch.Tensor,
    pas: torch.Tensor,
    pdp: torch.Tensor,
    pca: GroupedPCA,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    pas_values: list[np.ndarray] = []
    pdp_values: list[np.ndarray] = []
    base_pas_values: list[np.ndarray] = []
    base_pdp_values: list[np.ndarray] = []
    actions: list[float] = []
    for episode in episodes:
        prediction_pas, prediction_pdp, pas_gain, pdp_gain, query_ids, anchor_pas, anchor_pdp = run_episode(
            model, episode, state, position, pas, pdp, pca, device
        )
        pas_values.append(grouped_cosine(prediction_pas, pas[query_ids]).cpu().numpy())
        pdp_values.append(grouped_cosine(prediction_pdp, pdp[query_ids]).cpu().numpy())
        base_pas_values.append(grouped_cosine(anchor_pas, pas[query_ids]).cpu().numpy())
        base_pdp_values.append(grouped_cosine(anchor_pdp, pdp[query_ids]).cpu().numpy())
        actions.append(float(torch.cat([pas_gain.flatten(), pdp_gain.flatten()]).square().mean().sqrt().cpu()))
    pas_score = np.concatenate(pas_values)
    pdp_score = np.concatenate(pdp_values)
    base_pas = np.concatenate(base_pas_values)
    base_pdp = np.concatenate(base_pdp_values)
    score = 0.5 * (pas_score + pdp_score)
    base = 0.5 * (base_pas + base_pdp)
    return {
        "pas": float(np.mean(pas_score)),
        "pdp": float(np.mean(pdp_score)),
        "mean": float(np.mean(score)),
        "base_pas": float(np.mean(base_pas)),
        "base_pdp": float(np.mean(base_pdp)),
        "base_mean": float(np.mean(base)),
        "delta": float(np.mean(score - base)),
        "positive_fraction": float(np.mean(score >= base)),
        "query_count": int(len(score)),
        "log_gain_rms": float(np.mean(actions)),
    }


def train_fold(fold: int, config: Config, device: torch.device) -> dict[str, object]:
    set_seed(config.seed + fold * 1009)
    _, positions_numpy, test_positions, _, _ = load_positions()
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    folds = np.load(FOLD_CACHE)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    fit_valid = fit_ids[valid[fit_ids]]
    pas = torch.as_tensor(np.asarray(np.load(GROUPED_PAS, mmap_mode="r"), dtype=np.float32), device=device)
    pdp = torch.as_tensor(np.asarray(np.load(GROUPED_PDP, mmap_mode="r"), dtype=np.float32), device=device)
    pca = fit_grouped_pca(pas, pdp, fit_valid, config.seed + fold * 2003)
    train_state_raw, test_state_raw = load_state_features()
    train_state, _, feature_mean, feature_std = standardize_state(train_state_raw, test_state_raw, fit_ids)
    xy_mean = np.mean(positions_numpy[fit_ids, :2], axis=0).astype(np.float32)
    state = torch.as_tensor(train_state, dtype=torch.float32, device=device)
    position = torch.as_tensor(
        np.concatenate([positions_numpy - np.r_[xy_mean, 0.0], test_positions - np.r_[xy_mean, 0.0]], axis=0),
        dtype=torch.float32,
        device=device,
    )[: len(positions_numpy)]
    training = load_analog_episodes(fold, positions_numpy)
    held = load_held_episodes(fold, positions_numpy)
    model_config = ModelConfig(
        steps=config.steps,
        validation_interval=config.validation_interval,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        action_regularization=config.action_regularization,
        seed=config.seed,
    )
    model = GroupedAnchorHoleField(state.shape[1], model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    baseline = evaluate(model, held, state, position, pas, pdp, pca, device)
    history: list[dict[str, object]] = [{"step": 0, "validation": baseline}]
    best_score = float(baseline["mean"])
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    rng = np.random.default_rng(config.seed + fold * 3001)
    print(f"analog-hole fold={fold} holes={len(training)} step=0 validation={baseline}", flush=True)
    rolling: list[tuple[float, float, float]] = []
    for step in range(1, config.steps + 1):
        model.train()
        episode = training[int(rng.integers(0, len(training)))]
        prediction_pas, prediction_pdp, pas_gain, pdp_gain, query_ids, _, _ = run_episode(
            model, episode, state, position, pas, pdp, pca, device
        )
        pas_cosine = grouped_cosine(prediction_pas, pas[query_ids])
        pdp_cosine = grouped_cosine(prediction_pdp, pdp[query_ids])
        action = 0.5 * (pas_gain.square().mean() + pdp_gain.square().mean())
        loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean()) + config.action_regularization * action
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optimizer.step()
        rolling.append((float(pas_cosine.mean().detach().cpu()), float(pdp_cosine.mean().detach().cpu()), float(action.detach().sqrt().cpu())))
        if step % config.validation_interval == 0 or step == config.steps:
            validation = evaluate(model, held, state, position, pas, pdp, pca, device)
            train_summary = {
                "pas": float(np.mean([row[0] for row in rolling])),
                "pdp": float(np.mean([row[1] for row in rolling])),
                "log_gain_rms": float(np.mean([row[2] for row in rolling])),
            }
            history.append({"step": step, "train": train_summary, "validation": validation})
            if float(validation["mean"]) > best_score:
                best_score = float(validation["mean"])
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"analog-hole fold={fold} step={step}/{config.steps} "
                f"train={train_summary['pas']:.6f}/{train_summary['pdp']:.6f} "
                f"valid={validation['pas']:.6f}/{validation['pdp']:.6f}/"
                f"{validation['mean']:.6f} delta={validation['delta']:+.6f} "
                f"best={best_score:.6f}@{best_step}",
                flush=True,
            )
            rolling.clear()
    checkpoint = ARTIFACT_DIR / f"fold{fold}_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model_config": asdict(model_config),
            "train_config": asdict(config),
            "fold": fold,
            "best_step": best_step,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "xy_mean": xy_mean,
            "pca_mean": pca.mean.cpu(),
            "pca_vectors": pca.vectors.cpu(),
        },
        checkpoint,
    )
    return {
        "fold": fold,
        "analog_holes": int(len(training)),
        "analog_points": int(sum(len(item.query_ids) for item in training)),
        "baseline": baseline,
        "best_step": best_step,
        "best_score": best_score,
        "best_delta": float(best_score - float(baseline["base_mean"])),
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--validation-interval", type=int, default=Config.validation_interval)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = Config(steps=args.steps, validation_interval=args.validation_interval)
    device = torch.device(args.device)
    record = train_fold(args.fold, config, device)
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "official_inputs_only": True,
        "sionna_used": False,
        "primary_hypothesis": (
            "The identity-preserving grouped residual becomes transferable when training "
            "episodes use the same complete hole shape and shared boundary as the held query."
        ),
        "device": str(device),
        "config": asdict(config),
        "record": record,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
