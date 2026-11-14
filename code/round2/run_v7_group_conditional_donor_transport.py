#!/usr/bin/env python3
"""Learn low-rank group-specific donor transport from official Round 2 data."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META
from run_v3_anchored_neural_rbf import (
    ARTIFACT_DIR as V3_ARTIFACT_DIR,
    DESCRIPTORS,
    FOLD_CACHE,
    GEOMETRY,
    PAS_SIZE,
    AnchoredWeightCorrector,
    Config as V3Config,
    build_train_contexts,
    set_seed,
    standardize_geometry,
    tensor_context,
    validation_context,
)
from run_v3_grouped_link_refinement import (
    GROUPED_PAS,
    build_grouped_descriptors,
    stable_power_cosine,
)


VERSION = "Round2-E43-group-conditional-donor-transport"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "group_conditional_donor_transport"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_group_conditional_donor_transport"
REPORT = ANALYSIS_DIR / "group_conditional_donor_transport.json"
GEODESIC_CACHE = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
GROUPED_PDP_FINE = ARTIFACT_DIR / "grouped_pdp_fine_amplitude_f16.npy"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
TRAIN_PDP = PRE_DIR / "round2_train_pdp_link_support_f16.npy"

PAS_GROUPS = 32
PDP_GROUPS = 64
PAS_LENGTH = 256
PDP_LENGTH = 192
TINY = 1e-20


@dataclass(frozen=True)
class TransportConfig:
    rank: int = 6
    hidden: int = 128
    correction_l1_bound: float = 0.55
    correction_regularization: float = 0.018
    pas_smoothness: float = 0.003
    learning_rate: float = 7e-4
    weight_decay: float = 2e-5
    epochs: int = 10
    batch_size: int = 48
    seed: int = 20_260_806


def build_fine_pdp_descriptors() -> np.ndarray:
    if GROUPED_PDP_FINE.is_file():
        cached = np.load(GROUPED_PDP_FINE, mmap_mode="r")
        if cached.shape == (4000, PDP_GROUPS, PDP_LENGTH):
            return cached
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    source = np.load(TRAIN_PDP, mmap_mode="r")
    output = np.lib.format.open_memmap(
        GROUPED_PDP_FINE,
        mode="w+",
        dtype=np.float16,
        shape=(len(source), PDP_GROUPS, PDP_LENGTH),
    )
    for start in range(0, len(source), 16):
        end = min(start + 16, len(source))
        value = np.asarray(source[start:end], dtype=np.float32)
        value = value.reshape(end - start, 2, 4, 4, 2, 4, 4, 192)
        value = value.mean(axis=(3, 5), dtype=np.float64).astype(np.float32)
        value = value.reshape(end - start, PDP_GROUPS, PDP_LENGTH)
        output[start:end] = np.sqrt(np.maximum(value, 0.0)).astype(np.float16)
    output.flush()
    return np.load(GROUPED_PDP_FINE, mmap_mode="r")


def unit_amplitude(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(TINY)


def _numeric_pair_features(
    query_position: torch.Tensor,
    donor_position: torch.Tensor,
    distances: torch.Tensor,
    base_weights: torch.Tensor,
) -> torch.Tensor:
    position_delta = donor_position - query_position[:, None, :]
    direction = position_delta / distances[:, :, None].clamp_min(1e-4)
    rank = torch.linspace(
        0.0, 1.0, donor_position.shape[1], device=donor_position.device
    )[None, :, None].expand(len(query_position), -1, -1)
    return torch.cat(
        [
            position_delta / 20.0,
            distances[:, :, None] / 20.0,
            torch.log1p(distances[:, :, None]),
            direction,
            base_weights[:, :, None].clamp(-2.0, 2.0),
            base_weights.abs()[:, :, None].clamp(0.0, 2.0),
            rank,
        ],
        dim=-1,
    )


class GroupConditionalTransport(nn.Module):
    def __init__(
        self,
        geometry_size: int,
        geodesic_size: int,
        v3_config: V3Config,
        config: TransportConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.v3_config = v3_config
        self.base = AnchoredWeightCorrector(geometry_size, v3_config)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        pair_input = v3_config.hidden + geodesic_size + 5
        self.pair_adapter = nn.Sequential(
            nn.Linear(pair_input, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
        )
        self.pas_pair_rank = nn.Linear(config.hidden, config.rank)
        self.pdp_pair_rank = nn.Linear(config.hidden, config.rank)
        query_input = v3_config.hidden + geodesic_size
        self.pas_query_group = nn.Sequential(
            nn.Linear(query_input, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, PAS_GROUPS * config.rank),
        )
        self.pdp_query_group = nn.Sequential(
            nn.Linear(query_input, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, PDP_GROUPS * config.rank),
        )
        self.pas_similarity_scale = nn.Parameter(torch.zeros(PAS_GROUPS))
        self.pdp_similarity_scale = nn.Parameter(torch.zeros(PDP_GROUPS))
        self.pas_gate = nn.Sequential(
            nn.Linear(query_input, config.hidden // 2),
            nn.SiLU(),
            nn.Linear(config.hidden // 2, PAS_GROUPS),
        )
        self.pdp_gate = nn.Sequential(
            nn.Linear(query_input, config.hidden // 2),
            nn.SiLU(),
            nn.Linear(config.hidden // 2, PDP_GROUPS),
        )
        nn.init.zeros_(self.pas_pair_rank.weight)
        nn.init.zeros_(self.pas_pair_rank.bias)
        nn.init.zeros_(self.pdp_pair_rank.weight)
        nn.init.zeros_(self.pdp_pair_rank.bias)
        nn.init.zeros_(self.pas_query_group[-1].weight)
        nn.init.normal_(self.pas_query_group[-1].bias, std=0.03)
        nn.init.zeros_(self.pdp_query_group[-1].weight)
        nn.init.normal_(self.pdp_query_group[-1].bias, std=0.03)
        nn.init.zeros_(self.pas_gate[-1].weight)
        nn.init.constant_(self.pas_gate[-1].bias, -1.5)
        nn.init.zeros_(self.pdp_gate[-1].weight)
        nn.init.constant_(self.pdp_gate[-1].bias, -1.5)

    def load_base(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.base.load_state_dict(state_dict)

    def _base_state(
        self,
        query_geometry: torch.Tensor,
        query_position: torch.Tensor,
        donor_geometry: torch.Tensor,
        donor_position: torch.Tensor,
        donor_spectrum: torch.Tensor,
        distances: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            query_latent = self.base.query_encoder(query_geometry)
            donor_latent = self.base.spectral_encoder(donor_spectrum)
            geometry_delta = donor_geometry - query_geometry[:, None, :]
            numeric = _numeric_pair_features(
                query_position, donor_position, distances, base_weights
            )
            query_expanded = query_latent[:, None, :].expand(
                -1, donor_spectrum.shape[1], -1
            )
            pair = self.base.pair_encoder(
                torch.cat([query_expanded, geometry_delta, donor_latent, numeric], dim=-1)
            )
            pooled_mean = pair.mean(dim=1, keepdim=True).expand_as(pair)
            pooled_anchor = torch.sum(
                base_weights[:, :, None] * pair, dim=1, keepdim=True
            ).expand_as(pair)
            raw = torch.tanh(
                self.base.correction_head(
                    torch.cat([pair, pooled_mean, pooled_anchor], dim=-1)
                )
            )
            raw = raw - raw.mean(dim=1, keepdim=True)
            correction = self.v3_config.correction_l1_bound / raw.shape[1] * raw
            pas_weights = base_weights + correction[:, :, 0]
            pdp_weights = base_weights + correction[:, :, 1]
        return query_latent, pair, pas_weights, pdp_weights

    @staticmethod
    def _group_similarity(
        donors: torch.Tensor, anchor: torch.Tensor
    ) -> torch.Tensor:
        donor_power = donors.square()
        anchor_power = anchor.square()
        numerator = torch.sum(donor_power * anchor_power[:, None], dim=-1)
        denominator = torch.linalg.vector_norm(donor_power, dim=-1)
        denominator *= torch.linalg.vector_norm(anchor_power, dim=-1)[:, None]
        return numerator / denominator.clamp_min(TINY)

    def _metric_weights(
        self,
        pair_feature: torch.Tensor,
        query_feature: torch.Tensor,
        anchor_weights: torch.Tensor,
        donors: torch.Tensor,
        pair_head: nn.Linear,
        query_head: nn.Sequential,
        gate_head: nn.Sequential,
        similarity_scale: torch.Tensor,
        groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = unit_amplitude(
            torch.relu(torch.einsum("bk,bkgl->bgl", anchor_weights, donors))
        )
        similarity = self._group_similarity(donors, anchor)
        pair_rank = pair_head(pair_feature)
        query_rank = query_head(query_feature).reshape(
            len(query_feature), groups, self.config.rank
        )
        score = torch.einsum("bkr,bgr->bkg", pair_rank, query_rank)
        score += similarity * similarity_scale[None, None, :]
        score = torch.tanh(score)
        score = score - score.mean(dim=1, keepdim=True)
        gate = torch.sigmoid(gate_head(query_feature))
        correction = (
            self.config.correction_l1_bound
            / anchor_weights.shape[1]
            * score
            * gate[:, None, :]
        )
        weights = anchor_weights[:, :, None] + correction
        prediction = unit_amplitude(
            torch.relu(torch.einsum("bkg,bkgl->bgl", weights, donors))
        )
        return prediction, correction, gate

    def forward(
        self,
        query_geometry: torch.Tensor,
        query_geodesic: torch.Tensor,
        query_position: torch.Tensor,
        donor_geometry: torch.Tensor,
        donor_geodesic: torch.Tensor,
        donor_position: torch.Tensor,
        donor_spectrum: torch.Tensor,
        donor_pas: torch.Tensor,
        donor_pdp: torch.Tensor,
        distances: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        query_latent, pair, pas_anchor, pdp_anchor = self._base_state(
            query_geometry,
            query_position,
            donor_geometry,
            donor_position,
            donor_spectrum,
            distances,
            base_weights,
        )
        geodesic_delta = donor_geodesic - query_geodesic[:, None, :]
        route_numeric = torch.cat(
            [
                geodesic_delta,
                distances[:, :, None] / 20.0,
                torch.log1p(distances[:, :, None]),
                base_weights[:, :, None],
                pas_anchor[:, :, None] - pdp_anchor[:, :, None],
                0.5 * (pas_anchor[:, :, None] + pdp_anchor[:, :, None]),
            ],
            dim=-1,
        )
        pair_feature = self.pair_adapter(torch.cat([pair, route_numeric], dim=-1))
        query_feature = torch.cat([query_latent, query_geodesic], dim=1)
        pas, pas_correction, pas_gate = self._metric_weights(
            pair_feature,
            query_feature,
            pas_anchor,
            donor_pas,
            self.pas_pair_rank,
            self.pas_query_group,
            self.pas_gate,
            self.pas_similarity_scale,
            PAS_GROUPS,
        )
        pdp, pdp_correction, pdp_gate = self._metric_weights(
            pair_feature,
            query_feature,
            pdp_anchor,
            donor_pdp,
            self.pdp_pair_rank,
            self.pdp_query_group,
            self.pdp_gate,
            self.pdp_similarity_scale,
            PDP_GROUPS,
        )
        return pas, pdp, {
            "pas_correction": pas_correction,
            "pdp_correction": pdp_correction,
            "pas_gate": pas_gate,
            "pdp_gate": pdp_gate,
        }


def prepare_tensors(
    fit_ids: np.ndarray,
    base_checkpoint: dict[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor | np.ndarray]:
    _, positions_numpy, _, _, _ = load_positions()
    geometry_numpy = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)
    geometry_standard, _, _ = standardize_geometry(geometry_numpy, fit_ids)
    geodesic_numpy = np.load(GEODESIC_CACHE, allow_pickle=False)[
        "train_features"
    ].astype(np.float32)
    geodesic_mean = np.mean(geodesic_numpy[fit_ids], axis=0, dtype=np.float64).astype(
        np.float32
    )
    geodesic_std = np.std(geodesic_numpy[fit_ids], axis=0, dtype=np.float64).astype(
        np.float32
    )
    geodesic_std = np.maximum(geodesic_std, 1e-4)
    geodesic_standard = (geodesic_numpy - geodesic_mean) / geodesic_std
    descriptors = np.load(DESCRIPTORS, allow_pickle=False)
    spectrum_numpy = np.concatenate(
        [
            np.sqrt(np.maximum(descriptors["global_pas"].astype(np.float32), 0.0)),
            np.sqrt(np.maximum(descriptors["global_pdp"].astype(np.float32), 0.0)),
        ],
        axis=1,
    ).astype(np.float32)
    pas_numpy, _ = build_grouped_descriptors()
    pdp_numpy = build_fine_pdp_descriptors()
    return {
        "positions": torch.as_tensor(positions_numpy, dtype=torch.float32, device=device),
        "geometry": torch.as_tensor(geometry_standard, dtype=torch.float32, device=device),
        "geodesic": torch.as_tensor(geodesic_standard, dtype=torch.float32, device=device),
        "spectrum": torch.as_tensor(spectrum_numpy, dtype=torch.float32, device=device),
        "pas": torch.as_tensor(np.asarray(pas_numpy), dtype=torch.float32, device=device),
        "pdp": torch.as_tensor(np.asarray(pdp_numpy), dtype=torch.float32, device=device),
        "geodesic_mean": geodesic_mean,
        "geodesic_std": geodesic_std,
        "base_checkpoint": base_checkpoint,
    }


def forward_context(
    model: GroupConditionalTransport,
    context: dict[str, torch.Tensor],
    rows: torch.Tensor,
    guard: torch.Tensor | None,
    tensors: dict[str, torch.Tensor | np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    query_ids = context["query_ids"][rows]
    if context["donor_ids"].ndim == 3:
        if guard is None:
            raise ValueError("guard indices are required for training contexts")
        donor_ids = context["donor_ids"][rows, guard]
        weights = context["weights"][rows, guard]
        distances = context["distances"][rows, guard]
    else:
        donor_ids = context["donor_ids"][rows]
        weights = context["weights"][rows]
        distances = context["distances"][rows]
    geometry = tensors["geometry"]
    geodesic = tensors["geodesic"]
    positions = tensors["positions"]
    spectrum = tensors["spectrum"]
    pas = tensors["pas"]
    pdp = tensors["pdp"]
    if not all(isinstance(value, torch.Tensor) for value in (geometry, geodesic, positions, spectrum, pas, pdp)):
        raise TypeError("tensor cache contains non-tensor model inputs")
    prediction_pas, prediction_pdp, diagnostics = model(
        geometry[query_ids],
        geodesic[query_ids],
        positions[query_ids, :2],
        geometry[donor_ids],
        geodesic[donor_ids],
        positions[donor_ids, :2],
        spectrum[donor_ids],
        pas[donor_ids],
        pdp[donor_ids],
        distances,
        weights,
    )
    return prediction_pas, prediction_pdp, diagnostics, query_ids


@torch.no_grad()
def evaluate(
    model: GroupConditionalTransport,
    table,
    tensors: dict[str, torch.Tensor | np.ndarray],
    batch_size: int,
    return_predictions: bool = False,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray] | None]:
    model.eval()
    context = tensor_context(table, tensors["geometry"].device)
    pas_values = []
    pdp_values = []
    pas_gate = []
    pdp_gate = []
    pas_predictions = []
    pdp_predictions = []
    for start in range(0, len(table.query_ids), batch_size):
        end = min(start + batch_size, len(table.query_ids))
        rows = torch.arange(start, end, device=tensors["geometry"].device)
        predicted_pas, predicted_pdp, diagnostics, query_ids = forward_context(
            model, context, rows, None, tensors
        )
        pas_values.append(
            stable_power_cosine(predicted_pas, tensors["pas"][query_ids])
            .mean(dim=1)
            .cpu()
            .numpy()
        )
        pdp_values.append(
            stable_power_cosine(predicted_pdp, tensors["pdp"][query_ids])
            .mean(dim=1)
            .cpu()
            .numpy()
        )
        pas_gate.append(diagnostics["pas_gate"].mean(dim=1).cpu().numpy())
        pdp_gate.append(diagnostics["pdp_gate"].mean(dim=1).cpu().numpy())
        if return_predictions:
            pas_predictions.append(predicted_pas.cpu().numpy().astype(np.float16))
            pdp_predictions.append(predicted_pdp.cpu().numpy().astype(np.float16))
    pas_score = np.concatenate(pas_values)
    pdp_score = np.concatenate(pdp_values)
    payload = {
        "pas": float(np.mean(pas_score)),
        "pdp": float(np.mean(pdp_score)),
        "mean": float(np.mean(0.5 * (pas_score + pdp_score))),
        "pas_gate": float(np.mean(np.concatenate(pas_gate))),
        "pdp_gate": float(np.mean(np.concatenate(pdp_gate))),
    }
    predictions = None
    if return_predictions:
        predictions = (np.concatenate(pas_predictions), np.concatenate(pdp_predictions))
    return payload, predictions


def train_fold(
    fold: int,
    config: TransportConfig,
    device: torch.device,
) -> dict[str, object]:
    set_seed(config.seed + fold * 1009)
    base_checkpoint = torch.load(
        V3_ARTIFACT_DIR / f"fold{fold}_final.pt",
        map_location=device,
        weights_only=False,
    )
    v3_config = V3Config(**base_checkpoint["config"])
    _, positions_numpy, _, site, _ = load_positions()
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    valid = np.load(DESCRIPTORS, allow_pickle=False)["valid"].astype(bool)
    train_table = build_train_contexts(
        fold, positions_numpy, site, valid, fit_ids, v3_config
    )
    held_table = validation_context(fold, query_ids, fold_id)
    tensors = prepare_tensors(fit_ids, base_checkpoint, device)
    model = GroupConditionalTransport(
        tensors["geometry"].shape[1],
        tensors["geodesic"].shape[1],
        v3_config,
        config,
    ).to(device)
    model.load_base(base_checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_context = tensor_context(train_table, device)
    rng = np.random.default_rng(config.seed + fold * 1009)
    history = []
    baseline, baseline_prediction = evaluate(
        model, held_table, tensors, config.batch_size, return_predictions=True
    )
    history.append({"epoch": 0, "held": baseline})
    prediction_pas = [baseline_prediction[0]]
    prediction_pdp = [baseline_prediction[1]]
    best_score = baseline["mean"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    print(f"group-transport fold={fold} epoch=0 held={baseline}", flush=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(train_table.query_ids))
        train_rows = []
        for start in range(0, len(order), config.batch_size):
            local = order[start : start + config.batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=device)
            guard = torch.as_tensor(
                rng.integers(0, len(v3_config.guard_radii_m), size=len(local)),
                dtype=torch.long,
                device=device,
            )
            predicted_pas, predicted_pdp, diagnostics, query = forward_context(
                model, train_context, rows, guard, tensors
            )
            pas_cosine = stable_power_cosine(
                predicted_pas, tensors["pas"][query]
            ).mean(dim=1)
            pdp_cosine = stable_power_cosine(
                predicted_pdp, tensors["pdp"][query]
            ).mean(dim=1)
            regularization = config.correction_regularization * (
                diagnostics["pas_correction"].square().mean()
                + diagnostics["pdp_correction"].square().mean()
            ) * (v3_config.neighbors / max(config.correction_l1_bound**2, 1e-8))
            pas_correction = diagnostics["pas_correction"].reshape(
                len(local), v3_config.neighbors, 4, 8
            )
            smoothness = config.pas_smoothness * torch.mean(
                torch.square(pas_correction[:, :, :, 1:] - pas_correction[:, :, :, :-1])
            )
            loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean())
            loss = loss + regularization + smoothness
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.5
            )
            optimizer.step()
            train_rows.append(
                (
                    float(loss.detach().cpu()),
                    float(pas_cosine.mean().detach().cpu()),
                    float(pdp_cosine.mean().detach().cpu()),
                )
            )
        held, held_prediction = evaluate(
            model, held_table, tensors, config.batch_size, return_predictions=True
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean([row[0] for row in train_rows])),
                "train_pas": float(np.mean([row[1] for row in train_rows])),
                "train_pdp": float(np.mean([row[2] for row in train_rows])),
                "held": held,
            }
        )
        prediction_pas.append(held_prediction[0])
        prediction_pdp.append(held_prediction[1])
        if held["mean"] > best_score:
            best_score = held["mean"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"group-transport fold={fold} epoch={epoch}/{config.epochs} "
            f"train={history[-1]['train_pas']:.6f}/{history[-1]['train_pdp']:.6f} "
            f"held={held['pas']:.6f}/{held['pdp']:.6f}/{held['mean']:.6f} "
            f"best={best_score:.6f}@{best_epoch}",
            flush=True,
        )

    checkpoint_path = ARTIFACT_DIR / f"fold{fold}_best.pt"
    prediction_path = ARTIFACT_DIR / f"fold{fold}_held_predictions.npz"
    torch.save(
        {
            "state_dict": best_state,
            "base_state_dict": base_checkpoint["state_dict"],
            "v3_config": asdict(v3_config),
            "transport_config": asdict(config),
            "best_epoch": best_epoch,
            "geodesic_mean": tensors["geodesic_mean"],
            "geodesic_std": tensors["geodesic_std"],
        },
        checkpoint_path,
    )
    np.savez_compressed(
        prediction_path,
        query_ids=held_table.query_ids.astype(np.int32),
        pas=np.stack(prediction_pas, axis=0),
        pdp=np.stack(prediction_pdp, axis=0),
    )
    return {
        "fold": fold,
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
        "checkpoint": PATHS.relative(checkpoint_path),
        "held_predictions": PATHS.relative(prediction_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="3")
    parser.add_argument("--epochs", type=int, default=TransportConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TransportConfig.batch_size)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    build_grouped_descriptors()
    build_fine_pdp_descriptors()
    config = TransportConfig(epochs=args.epochs, batch_size=args.batch_size)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    records = [train_fold(fold, config, torch.device(args.device)) for fold in folds]
    payload = {
        "experiment": VERSION,
        "status": "analysis_only",
        "official_inputs_only": True,
        "sionna_used": False,
        "primary_hypothesis": (
            "The dominant remaining spectral error comes from forcing all official PAS/PDP "
            "groups to share one donor mixture; low-rank group-specific donor transport can "
            "resolve path-state ambiguity while keeping V3's stable spatial anchor."
        ),
        "config": asdict(config),
        "folds": folds,
        "records": records,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
