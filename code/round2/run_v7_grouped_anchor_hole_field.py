#!/usr/bin/env python3
"""Train an exact-identity whole-hole residual model for grouped PAS/PDP."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from data import load_positions
from project_paths import PATHS
from run_v3_anchored_neural_rbf import (
    Config as ContextConfig,
    FOLD_CACHE,
    OOF_META,
    build_train_contexts,
    set_seed,
    validation_context,
)
from run_v3_grouped_link_refinement import GROUPED_PAS, GROUPED_PDP
from run_v4_group_conditioned_link_field import load_state_features, standardize_state
from run_v4_masked_hole_neural_process import (
    CrossBlock,
    Episode,
    EpisodeBuilder,
    positional_features,
    validation_episodes,
)


VERSION = "Round2-E53-grouped-anchor-hole-field"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "grouped_anchor_hole_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_grouped_anchor_hole_field"
REPORT = ANALYSIS_DIR / "grouped_anchor_hole_field.json"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
PAS_GROUPS = 32
PAS_BINS = 256
PDP_GROUPS = 8
PDP_BINS = 192
PAS_SIZE = PAS_GROUPS * PAS_BINS
PDP_SIZE = PDP_GROUPS * PDP_BINS
OUTPUT_SIZE = PAS_SIZE + PDP_SIZE
LATENT = 128
TINY = 1e-20


@dataclass(frozen=True)
class Config:
    hidden: int = 192
    heads: int = 6
    self_layers: int = 2
    cross_layers: int = 2
    dropout: float = 0.05
    log_gain_bound: float = 0.45
    steps: int = 300
    validation_interval: int = 50
    learning_rate: float = 1.5e-4
    weight_decay: float = 3e-5
    action_regularization: float = 0.004
    seed: int = 20_260_806


@dataclass(frozen=True)
class GroupedPCA:
    mean: torch.Tensor
    vectors: torch.Tensor
    coefficients: torch.Tensor


def normalize_groups(value: torch.Tensor) -> torch.Tensor:
    value = torch.relu(value)
    return value / torch.sqrt(value.square().sum(dim=-1, keepdim=True).clamp_min(TINY))


def grouped_cosine(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    prediction_power = prediction.square()
    truth_power = truth.square()
    numerator = torch.sum(prediction_power * truth_power, dim=-1)
    product = torch.sum(prediction_power.square(), dim=-1) * torch.sum(
        truth_power.square(), dim=-1
    )
    return (numerator / torch.sqrt(product.clamp_min(TINY))).mean(dim=1)


def fit_grouped_pca(
    pas: torch.Tensor,
    pdp: torch.Tensor,
    fit_ids: np.ndarray,
    seed: int,
) -> GroupedPCA:
    torch.manual_seed(seed)
    matrix = torch.cat(
        [pas.flatten(1), pdp.flatten(1)], dim=1
    )
    selected = matrix[torch.as_tensor(fit_ids, dtype=torch.long, device=matrix.device)]
    mean = selected.mean(dim=0)
    _, _, vectors = torch.pca_lowrank(
        selected - mean,
        q=LATENT,
        center=False,
        niter=4,
    )
    coefficients = (matrix - mean) @ vectors
    return GroupedPCA(mean=mean, vectors=vectors, coefficients=coefficients)


def encode_grouped(
    pas: torch.Tensor,
    pdp: torch.Tensor,
    pca: GroupedPCA,
) -> torch.Tensor:
    return (torch.cat([pas.flatten(1), pdp.flatten(1)], dim=1) - pca.mean) @ pca.vectors


class GroupedAnchorHoleField(nn.Module):
    def __init__(self, state_size: int, config: Config) -> None:
        super().__init__()
        self.config = config
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size + 18, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
        )
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(LATENT, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(config.hidden * 2, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
        )
        self.query_fusion = nn.Sequential(
            nn.Linear(config.hidden * 2, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=config.heads,
            dim_feedforward=config.hidden * 3,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_self = nn.TransformerEncoder(
            layer, num_layers=config.self_layers, enable_nested_tensor=False
        )
        self.cross = nn.ModuleList(
            [CrossBlock(config.hidden, config.heads, config.dropout) for _ in range(config.cross_layers)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, OUTPUT_SIZE),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _tokens(
        self,
        state: torch.Tensor,
        position: torch.Tensor,
        coefficients: torch.Tensor,
        query: bool,
    ) -> torch.Tensor:
        geometry = self.state_encoder(torch.cat([state, positional_features(position)], dim=1))
        spectrum = self.spectrum_encoder(coefficients)
        fusion = self.query_fusion if query else self.context_fusion
        return fusion(torch.cat([geometry, spectrum], dim=1))

    def forward(
        self,
        query_state: torch.Tensor,
        query_position: torch.Tensor,
        anchor_coefficients: torch.Tensor,
        context_state: torch.Tensor,
        context_position: torch.Tensor,
        context_coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._tokens(query_state, query_position, anchor_coefficients, True)[None]
        context = self._tokens(context_state, context_position, context_coefficients, False)[None]
        query = self.query_self(query)
        for block in self.cross:
            query = block(query, context)
        raw = self.config.log_gain_bound * torch.tanh(self.output(query[0]))
        return (
            raw[:, :PAS_SIZE].reshape(-1, PAS_GROUPS, PAS_BINS),
            raw[:, PAS_SIZE:].reshape(-1, PDP_GROUPS, PDP_BINS),
        )


def anchor_grouped(
    value: torch.Tensor,
    donor_ids: np.ndarray,
    donor_weights: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    ids = torch.as_tensor(donor_ids, dtype=torch.long, device=device)
    weights = torch.as_tensor(donor_weights, dtype=torch.float32, device=device)
    return normalize_groups(torch.einsum("qk,qkgd->qgd", weights, value[ids]))


def run_episode(
    model: GroupedAnchorHoleField,
    episode: Episode,
    table,
    state: torch.Tensor,
    position: torch.Tensor,
    pas: torch.Tensor,
    pdp: torch.Tensor,
    pca: GroupedPCA,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_ids = torch.as_tensor(table.query_ids[episode.query_rows], dtype=torch.long, device=device)
    context_ids = torch.as_tensor(episode.context_ids, dtype=torch.long, device=device)
    anchor_pas = anchor_grouped(pas, episode.donor_ids, episode.donor_weights, device)
    anchor_pdp = anchor_grouped(pdp, episode.donor_ids, episode.donor_weights, device)
    anchor_coefficients = encode_grouped(anchor_pas, anchor_pdp, pca)
    pas_gain, pdp_gain = model(
        state[query_ids],
        position[query_ids, :2],
        anchor_coefficients,
        state[context_ids],
        position[context_ids, :2],
        pca.coefficients[context_ids],
    )
    prediction_pas = normalize_groups(anchor_pas * torch.exp(pas_gain))
    prediction_pdp = normalize_groups(anchor_pdp * torch.exp(pdp_gain))
    return prediction_pas, prediction_pdp, pas_gain, pdp_gain, query_ids, torch.cat(
        [grouped_cosine(anchor_pas, pas[query_ids])[:, None], grouped_cosine(anchor_pdp, pdp[query_ids])[:, None]],
        dim=1,
    )


@torch.no_grad()
def evaluate(
    model: GroupedAnchorHoleField,
    episodes: list[Episode],
    table,
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
    base_values: list[np.ndarray] = []
    action_values: list[float] = []
    for episode in episodes:
        pred_pas, pred_pdp, pas_gain, pdp_gain, query_ids, base = run_episode(
            model, episode, table, state, position, pas, pdp, pca, device
        )
        pas_values.append(grouped_cosine(pred_pas, pas[query_ids]).cpu().numpy())
        pdp_values.append(grouped_cosine(pred_pdp, pdp[query_ids]).cpu().numpy())
        base_values.append(base.cpu().numpy())
        action_values.append(float(torch.cat([pas_gain.flatten(), pdp_gain.flatten()]).square().mean().sqrt().cpu()))
    pas_score = np.concatenate(pas_values)
    pdp_score = np.concatenate(pdp_values)
    base = np.concatenate(base_values)
    score = 0.5 * (pas_score + pdp_score)
    base_score = 0.5 * (base[:, 0] + base[:, 1])
    return {
        "pas": float(np.mean(pas_score)),
        "pdp": float(np.mean(pdp_score)),
        "mean": float(np.mean(score)),
        "base_pas": float(np.mean(base[:, 0])),
        "base_pdp": float(np.mean(base[:, 1])),
        "base_mean": float(np.mean(base_score)),
        "delta": float(np.mean(score - base_score)),
        "positive_fraction": float(np.mean(score >= base_score)),
        "query_count": int(len(score)),
        "log_gain_rms": float(np.mean(action_values)),
    }


def train_fold(fold: int, config: Config, device: torch.device) -> dict[str, object]:
    set_seed(config.seed + fold * 1009)
    _, positions_numpy, test_positions, site, _ = load_positions()
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    folds = np.load(FOLD_CACHE)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    fit_valid = fit_ids[valid[fit_ids]]
    pas = torch.as_tensor(np.asarray(np.load(GROUPED_PAS, mmap_mode="r"), dtype=np.float32), device=device)
    pdp = torch.as_tensor(np.asarray(np.load(GROUPED_PDP, mmap_mode="r"), dtype=np.float32), device=device)
    pca = fit_grouped_pca(pas, pdp, fit_valid, config.seed + fold * 2003)
    metadata = np.load(OOF_META)
    oof_query_ids = metadata["query_ids"].astype(np.int64)
    oof_fold_id = metadata["fold_id"].astype(np.int8)
    context_config = ContextConfig()
    train_table = build_train_contexts(fold, positions_numpy, site, valid, fit_ids, context_config)
    held_table = validation_context(fold, oof_query_ids, oof_fold_id)
    train_state_raw, test_state_raw = load_state_features()
    train_state, _, feature_mean, feature_std = standardize_state(train_state_raw, test_state_raw, fit_ids)
    xy_mean = np.mean(positions_numpy[fit_ids, :2], axis=0).astype(np.float32)
    state = torch.as_tensor(train_state, dtype=torch.float32, device=device)
    position = torch.as_tensor(
        np.concatenate([positions_numpy - np.r_[xy_mean, 0.0], test_positions - np.r_[xy_mean, 0.0]], axis=0),
        dtype=torch.float32,
        device=device,
    )[: len(positions_numpy)]
    builder = EpisodeBuilder(train_table, positions_numpy, site, config.seed + fold * 3001)
    held_episodes = validation_episodes(held_table, site)
    model = GroupedAnchorHoleField(state.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    baseline = evaluate(model, held_episodes, held_table, state, position, pas, pdp, pca, device)
    history: list[dict[str, object]] = [{"step": 0, "validation": baseline}]
    best_score = float(baseline["mean"])
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    print(f"group-hole fold={fold} step=0 validation={baseline}", flush=True)
    rolling: list[tuple[float, float, float, float]] = []
    for step in range(1, config.steps + 1):
        model.train()
        episode = builder.sample()
        pred_pas, pred_pdp, pas_gain, pdp_gain, query_ids, _ = run_episode(
            model, episode, train_table, state, position, pas, pdp, pca, device
        )
        pas_cosine = grouped_cosine(pred_pas, pas[query_ids])
        pdp_cosine = grouped_cosine(pred_pdp, pdp[query_ids])
        action = 0.5 * (pas_gain.square().mean() + pdp_gain.square().mean())
        loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean())
        loss = loss + config.action_regularization * action
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optimizer.step()
        values = (
            float(loss.detach().cpu()),
            float(pas_cosine.mean().detach().cpu()),
            float(pdp_cosine.mean().detach().cpu()),
            float(action.detach().sqrt().cpu()),
        )
        if not np.isfinite(values[0]):
            raise FloatingPointError("non-finite grouped-hole loss")
        rolling.append(values)
        if step % config.validation_interval == 0 or step == config.steps:
            validation = evaluate(model, held_episodes, held_table, state, position, pas, pdp, pca, device)
            train_summary = {
                "loss": float(np.mean([row[0] for row in rolling])),
                "pas": float(np.mean([row[1] for row in rolling])),
                "pdp": float(np.mean([row[2] for row in rolling])),
                "log_gain_rms": float(np.mean([row[3] for row in rolling])),
            }
            history.append({"step": step, "train": train_summary, "validation": validation})
            if float(validation["mean"]) > best_score:
                best_score = float(validation["mean"])
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"group-hole fold={fold} step={step}/{config.steps} "
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
            "config": asdict(config),
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
        "fit_count": int(len(fit_ids)),
        "fit_valid_count": int(len(fit_valid)),
        "baseline": baseline,
        "best_step": best_step,
        "best_score": best_score,
        "best_delta": float(best_score - float(baseline["base_mean"])),
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--validation-interval", type=int, default=Config.validation_interval)
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
            "Whole-hole context can predict exact-identity multiplicative residuals "
            "for the 32 PAS and 8 PDP groups used by the V4 target pipeline."
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
