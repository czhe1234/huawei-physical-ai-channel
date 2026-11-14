#!/usr/bin/env python3
"""Train an attentive neural process on complete masked spatial holes."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
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
from run_v4_group_conditioned_link_field import load_state_features, standardize_state


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_masked_hole_neural_process"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_masked_hole_neural_process"
REPORT = ANALYSIS_DIR / "masked_hole_neural_process.json"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"

PCA_COMPONENTS = 128
TRAIN_GUARD_INDEX = 6
CONTEXT_DONORS_PER_QUERY = 16
MAX_CONTEXT_POINTS = 384
HOLE_SIZES = (24, 32, 48, 64, 80)


@dataclass(frozen=True)
class Config:
    hidden: int = 192
    heads: int = 6
    self_layers: int = 2
    cross_layers: int = 2
    dropout: float = 0.05
    delta_bound: float = 0.75
    steps: int = 1600
    validation_interval: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 2e-5
    coefficient_weight: float = 0.02
    action_regularization: float = 0.002
    seed: int = 20_260_805


def normalize_amplitude(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    norm = np.sqrt(np.sum(output * output, axis=1, keepdims=True, dtype=np.float64))
    return np.divide(
        output,
        np.maximum(norm, 1e-20),
        out=np.zeros_like(output),
        where=norm > 0.0,
    )


def power_cosine_torch(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    predicted_power = predicted.square()
    truth_power = truth.square()
    numerator = torch.sum(predicted_power * truth_power, dim=1)
    product = torch.sum(predicted_power.square(), dim=1) * torch.sum(
        truth_power.square(), dim=1
    )
    return numerator / torch.sqrt(product.clamp_min(1e-20))


def power_cosine_numpy(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    predicted_power = np.square(predicted)
    truth_power = np.square(truth)
    numerator = np.sum(predicted_power * truth_power, axis=1, dtype=np.float64)
    product = np.sum(predicted_power**2, axis=1, dtype=np.float64) * np.sum(
        truth_power**2, axis=1, dtype=np.float64
    )
    return np.divide(
        numerator,
        np.sqrt(np.maximum(product, 1e-30)),
        out=np.zeros_like(numerator),
        where=product > 0.0,
    )


class SpectralPCA:
    def __init__(self, amplitude: np.ndarray, fit_ids: np.ndarray) -> None:
        self.pas_size = 256
        pca = PCA(n_components=PCA_COMPONENTS, random_state=20_260_805)
        pca.fit(amplitude[fit_ids])
        self.mean = pca.mean_.astype(np.float32)
        self.components = pca.components_.astype(np.float32)
        self.scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-8)).astype(np.float32)
        self.coefficients = (
            pca.transform(amplitude).astype(np.float32) / self.scale[None]
        )

    def torch_constants(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "mean": torch.as_tensor(self.mean, device=device),
            "components": torch.as_tensor(self.components, device=device),
            "scale": torch.as_tensor(self.scale, device=device),
        }

    @staticmethod
    def encode(value: torch.Tensor, constants: dict[str, torch.Tensor]) -> torch.Tensor:
        return ((value - constants["mean"]) @ constants["components"].T) / constants[
            "scale"
        ]

    @staticmethod
    def decode(value: torch.Tensor, constants: dict[str, torch.Tensor]) -> torch.Tensor:
        amplitude = constants["mean"] + (
            value * constants["scale"]
        ) @ constants["components"]
        pas = torch.relu(amplitude[:, :256])
        pdp = torch.relu(amplitude[:, 256:])
        pas = pas / torch.sqrt(pas.square().sum(dim=1, keepdim=True).clamp_min(1e-20))
        pdp = pdp / torch.sqrt(pdp.square().sum(dim=1, keepdim=True).clamp_min(1e-20))
        return torch.cat([pas, pdp], dim=1)


def positional_features(position: torch.Tensor) -> torch.Tensor:
    scaled = position / 20.0
    outputs = [scaled]
    for frequency in (0.5, 1.0, 2.0, 4.0):
        angle = position * (frequency * math.pi / 20.0)
        outputs.extend([torch.sin(angle), torch.cos(angle)])
    return torch.cat(outputs, dim=1)


class CrossBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.context_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.feed_norm = nn.LayerNorm(hidden)
        self.feed = nn.Sequential(
            nn.Linear(hidden, hidden * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        update, _ = self.attention(
            normalized_query,
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        query = query + update
        return query + self.feed(self.feed_norm(query))


class MaskedHoleNeuralProcess(nn.Module):
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
            nn.Linear(PCA_COMPONENTS, config.hidden),
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
            [
                CrossBlock(config.hidden, config.heads, config.dropout)
                for _ in range(config.cross_layers)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, PCA_COMPONENTS),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def tokens(
        self,
        state: torch.Tensor,
        position: torch.Tensor,
        coefficients: torch.Tensor,
        query: bool,
    ) -> torch.Tensor:
        geometry = self.state_encoder(
            torch.cat([state, positional_features(position)], dim=1)
        )
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
        query = self.tokens(
            query_state, query_position, anchor_coefficients, query=True
        )[None]
        context = self.tokens(
            context_state, context_position, context_coefficients, query=False
        )[None]
        query = self.query_self(query)
        for block in self.cross:
            query = block(query, context)
        delta = self.config.delta_bound * torch.tanh(self.output(query[0]))
        return anchor_coefficients + delta, delta


@dataclass(frozen=True)
class Episode:
    query_rows: np.ndarray
    donor_ids: np.ndarray
    donor_weights: np.ndarray
    context_ids: np.ndarray


class EpisodeBuilder:
    def __init__(
        self,
        table,
        positions: np.ndarray,
        site: np.ndarray,
        seed: int,
    ) -> None:
        self.table = table
        self.positions = positions
        self.site = site
        self.rng = np.random.default_rng(seed)
        self.query_ids = table.query_ids.astype(np.int64)
        self.row_lookup = {int(point): row for row, point in enumerate(self.query_ids)}
        self.site_rows = {
            int(value): np.flatnonzero(site[self.query_ids] == value)
            for value in np.unique(site[self.query_ids])
        }
        self.site_trees = {
            value: cKDTree(positions[self.query_ids[rows], :2])
            for value, rows in self.site_rows.items()
        }

    def context_from_donors(
        self,
        query_ids: np.ndarray,
        donor_ids: np.ndarray,
    ) -> np.ndarray:
        candidates = np.unique(donor_ids[:, :CONTEXT_DONORS_PER_QUERY])
        candidates = candidates[~np.isin(candidates, query_ids)]
        if len(candidates) <= MAX_CONTEXT_POINTS:
            return candidates.astype(np.int64)
        distance = cKDTree(self.positions[query_ids, :2]).query(
            self.positions[candidates, :2], k=1, workers=-1
        )[0]
        return candidates[np.argsort(distance)[:MAX_CONTEXT_POINTS]].astype(np.int64)

    def sample(self) -> Episode:
        site_value = int(self.rng.choice(list(self.site_rows)))
        rows = self.site_rows[site_value]
        size = min(int(self.rng.choice(HOLE_SIZES)), len(rows))
        center_local = int(self.rng.integers(0, len(rows)))
        local = np.asarray(
            self.site_trees[site_value].query(
                self.positions[self.query_ids[rows[center_local]], :2], k=size
            )[1]
        ).reshape(-1)
        query_rows = rows[local]
        query_ids = self.query_ids[query_rows]
        donor_ids = self.table.donor_ids[query_rows, TRAIN_GUARD_INDEX].astype(np.int64)
        donor_weights = self.table.weights[query_rows, TRAIN_GUARD_INDEX].astype(np.float32)
        overlap = np.isin(donor_ids, query_ids)
        if np.any(overlap):
            donor_weights = donor_weights.copy()
            donor_weights[overlap] = 0.0
            total = donor_weights.sum(axis=1, keepdims=True)
            donor_weights = np.divide(
                donor_weights,
                np.maximum(total, 1e-8),
                out=np.zeros_like(donor_weights),
                where=np.abs(total) > 1e-8,
            )
        context_ids = self.context_from_donors(query_ids, donor_ids)
        return Episode(query_rows, donor_ids, donor_weights, context_ids)


def validation_episodes(table, site: np.ndarray) -> list[Episode]:
    output: list[Episode] = []
    for site_value in np.unique(site[table.query_ids]):
        rows = np.flatnonzero(site[table.query_ids] == site_value)
        donor_ids = table.donor_ids[rows].astype(np.int64)
        donor_weights = table.weights[rows].astype(np.float32)
        candidates = np.unique(donor_ids[:, :CONTEXT_DONORS_PER_QUERY])
        if len(candidates) > MAX_CONTEXT_POINTS:
            _, positions, _, _, _ = load_positions()
            distance = cKDTree(positions[table.query_ids[rows], :2]).query(
                positions[candidates, :2], k=1, workers=-1
            )[0]
            candidates = candidates[np.argsort(distance)[:MAX_CONTEXT_POINTS]]
        output.append(Episode(rows, donor_ids, donor_weights, candidates.astype(np.int64)))
    return output


def anchor_amplitude(
    amplitude: torch.Tensor,
    donor_ids: np.ndarray,
    donor_weights: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    ids = torch.as_tensor(donor_ids, dtype=torch.long, device=device)
    weights = torch.as_tensor(donor_weights, dtype=torch.float32, device=device)
    value = torch.relu(torch.einsum("qk,qkd->qd", weights, amplitude[ids]))
    pas = value[:, :256]
    pdp = value[:, 256:]
    pas = pas / torch.sqrt(pas.square().sum(dim=1, keepdim=True).clamp_min(1e-20))
    pdp = pdp / torch.sqrt(pdp.square().sum(dim=1, keepdim=True).clamp_min(1e-20))
    return torch.cat([pas, pdp], dim=1)


def run_episode(
    model: MaskedHoleNeuralProcess,
    episode: Episode,
    table,
    state: torch.Tensor,
    position: torch.Tensor,
    amplitude: torch.Tensor,
    coefficients: torch.Tensor,
    pca_constants: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_ids = torch.as_tensor(
        table.query_ids[episode.query_rows], dtype=torch.long, device=device
    )
    context_ids = torch.as_tensor(episode.context_ids, dtype=torch.long, device=device)
    anchor = anchor_amplitude(
        amplitude, episode.donor_ids, episode.donor_weights, device
    )
    anchor_coefficients = SpectralPCA.encode(anchor, pca_constants)
    predicted_coefficients, delta = model(
        state[query_ids],
        position[query_ids, :2],
        anchor_coefficients,
        state[context_ids],
        position[context_ids, :2],
        coefficients[context_ids],
    )
    prediction = SpectralPCA.decode(predicted_coefficients, pca_constants)
    return prediction, predicted_coefficients, delta, query_ids, anchor


@torch.no_grad()
def evaluate(
    model: MaskedHoleNeuralProcess,
    episodes: list[Episode],
    table,
    state: torch.Tensor,
    position: torch.Tensor,
    amplitude: torch.Tensor,
    coefficients: torch.Tensor,
    pca_constants: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    pas_values: list[np.ndarray] = []
    pdp_values: list[np.ndarray] = []
    base_pas_values: list[np.ndarray] = []
    base_pdp_values: list[np.ndarray] = []
    delta_values: list[float] = []
    for episode in episodes:
        prediction, _, delta, query_ids, anchor = run_episode(
            model,
            episode,
            table,
            state,
            position,
            amplitude,
            coefficients,
            pca_constants,
            device,
        )
        truth = amplitude[query_ids]
        pas_values.append(
            power_cosine_torch(prediction[:, :256], truth[:, :256]).cpu().numpy()
        )
        pdp_values.append(
            power_cosine_torch(prediction[:, 256:], truth[:, 256:]).cpu().numpy()
        )
        base_pas_values.append(
            power_cosine_torch(anchor[:, :256], truth[:, :256]).cpu().numpy()
        )
        base_pdp_values.append(
            power_cosine_torch(anchor[:, 256:], truth[:, 256:]).cpu().numpy()
        )
        delta_values.append(float(delta.square().mean().sqrt().cpu()))
    pas = np.concatenate(pas_values)
    pdp = np.concatenate(pdp_values)
    base_pas = np.concatenate(base_pas_values)
    base_pdp = np.concatenate(base_pdp_values)
    return {
        "pas": float(np.mean(pas)),
        "pdp": float(np.mean(pdp)),
        "mean": float(np.mean(0.5 * (pas + pdp))),
        "base_pas": float(np.mean(base_pas)),
        "base_pdp": float(np.mean(base_pdp)),
        "base_mean": float(np.mean(0.5 * (base_pas + base_pdp))),
        "delta": float(np.mean(0.5 * (pas + pdp) - 0.5 * (base_pas + base_pdp))),
        "query_count": int(len(pas)),
        "coefficient_delta_rms": float(np.mean(delta_values)),
    }


def train_fold(fold: int, config: Config, device: torch.device) -> dict[str, object]:
    set_seed(config.seed + fold * 1009)
    _, positions_numpy, test_positions, site, _ = load_positions()
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    pas = normalize_amplitude(np.sqrt(np.maximum(descriptors["global_pas"], 0.0)))
    pdp = normalize_amplitude(np.sqrt(np.maximum(descriptors["global_pdp"], 0.0)))
    amplitude_numpy = np.concatenate([pas, pdp], axis=1).astype(np.float32)
    folds = np.load(FOLD_CACHE)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    fit_valid = fit_ids[valid[fit_ids]]
    pca = SpectralPCA(amplitude_numpy, fit_valid)
    metadata = np.load(OOF_META)
    oof_query_ids = metadata["query_ids"].astype(np.int64)
    oof_fold_id = metadata["fold_id"].astype(np.int8)
    context_config = ContextConfig()
    train_table = build_train_contexts(
        fold, positions_numpy, site, valid, fit_ids, context_config
    )
    held_table = validation_context(fold, oof_query_ids, oof_fold_id)
    train_state_raw, test_state_raw = load_state_features()
    train_state, _, feature_mean, feature_std = standardize_state(
        train_state_raw, test_state_raw, fit_ids
    )
    xy_mean = np.mean(positions_numpy[fit_ids, :2], axis=0).astype(np.float32)
    state = torch.as_tensor(train_state, dtype=torch.float32, device=device)
    position = torch.as_tensor(
        np.concatenate(
            [positions_numpy - np.r_[xy_mean, 0.0], test_positions - np.r_[xy_mean, 0.0]],
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    )[: len(positions_numpy)]
    amplitude = torch.as_tensor(amplitude_numpy, dtype=torch.float32, device=device)
    coefficients = torch.as_tensor(pca.coefficients, dtype=torch.float32, device=device)
    pca_constants = pca.torch_constants(device)
    builder = EpisodeBuilder(
        train_table, positions_numpy, site, config.seed + fold * 1009
    )
    held_episodes = validation_episodes(held_table, site)
    model = MaskedHoleNeuralProcess(state.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    baseline = evaluate(
        model,
        held_episodes,
        held_table,
        state,
        position,
        amplitude,
        coefficients,
        pca_constants,
        device,
    )
    history: list[dict[str, object]] = [{"step": 0, "validation": baseline}]
    best_score = baseline["mean"]
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    rng = np.random.default_rng(config.seed + fold * 2003)
    print(f"masked-hole fold={fold} step=0 validation={baseline}", flush=True)

    model.train()
    rolling: list[tuple[float, float, float]] = []
    for step in range(1, config.steps + 1):
        episode = builder.sample()
        prediction, predicted_coefficients, delta, query_ids, _ = run_episode(
            model,
            episode,
            train_table,
            state,
            position,
            amplitude,
            coefficients,
            pca_constants,
            device,
        )
        truth = amplitude[query_ids]
        pas_cosine = power_cosine_torch(prediction[:, :256], truth[:, :256])
        pdp_cosine = power_cosine_torch(prediction[:, 256:], truth[:, 256:])
        coefficient_loss = (predicted_coefficients - coefficients[query_ids]).square().mean()
        loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean())
        loss = loss + config.coefficient_weight * coefficient_loss
        loss = loss + config.action_regularization * delta.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optimizer.step()
        rolling.append(
            (
                float(loss.detach().cpu()),
                float(pas_cosine.mean().detach().cpu()),
                float(pdp_cosine.mean().detach().cpu()),
            )
        )
        if step % config.validation_interval == 0 or step == config.steps:
            validation = evaluate(
                model,
                held_episodes,
                held_table,
                state,
                position,
                amplitude,
                coefficients,
                pca_constants,
                device,
            )
            train_summary = {
                "loss": float(np.mean([row[0] for row in rolling])),
                "pas": float(np.mean([row[1] for row in rolling])),
                "pdp": float(np.mean([row[2] for row in rolling])),
            }
            history.append(
                {"step": step, "train": train_summary, "validation": validation}
            )
            if validation["mean"] > best_score:
                best_score = validation["mean"]
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"masked-hole fold={fold} step={step}/{config.steps} "
                f"train={train_summary['pas']:.6f}/{train_summary['pdp']:.6f} "
                f"valid={validation['pas']:.6f}/{validation['pdp']:.6f}/"
                f"{validation['mean']:.6f} delta={validation['delta']:+.6f} "
                f"best={best_score:.6f}@{best_step}",
                flush=True,
            )
            rolling.clear()
            model.train()
        if not np.isfinite(rolling[-1][0]) if rolling else False:
            raise FloatingPointError("non-finite masked-hole loss")

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
            "pca_mean": pca.mean,
            "pca_components": pca.components,
            "pca_scale": pca.scale,
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
        "best_delta": float(best_score - baseline["base_mean"]),
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--validation-interval", type=int, default=Config.validation_interval)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = Config(
        steps=args.steps, validation_interval=args.validation_interval
    )
    device = torch.device(args.device)
    record = train_fold(args.fold, config, device)
    payload = {
        "experiment": "Round2-E15-masked-hole-neural-process",
        "status": "analysis_only",
        "primary_hypothesis": (
            "An attentive neural process trained on complete masked holes can combine "
            "boundary truth, V2 query anchors and query-query relations to infer path-state "
            "transitions that independent interpolation cannot identify."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "test_test_used": True,
        "device": str(device),
        "config": asdict(config),
        "record": record,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
