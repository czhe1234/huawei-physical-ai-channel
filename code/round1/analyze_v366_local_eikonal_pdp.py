#!/usr/bin/env python3
"""Query-local eikonal delay-surface model for normalized official PDP.

The path identity in this experiment exists only inside one query neighborhood.
Continuous donor components are grouped when their delay can be explained by a
locally affine path-length surface whose gradient obeys the physical eikonal
bound. No global virtual-source identity is assigned.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed, parallel_config
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v365_multiview_component_pdp as v365
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 366
VERSION = "v366_local_eikonal_pdp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
AUDIT_CACHE = OUT_DIR / "v366_eikonal_hypothesis_audit.npz"
OOF_OUT = OUT_DIR / "v366_local_eikonal_pdp_oof.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

N_DELAY = 192
COMPONENT_COUNT = 12
NEIGHBORS = 32
HYPOTHESES = 64
MIN_SUPPORT = 4
SPEED_OF_LIGHT = 299_792_458.0
SUBCARRIER_SPACING = 30_000.0
THEORETICAL_GRADIENT = N_DELAY * SUBCARRIER_SPACING / SPEED_OF_LIGHT
MAX_GRADIENT = 1.15 * THEORETICAL_GRADIENT
INITIAL_TOLERANCE = 0.12
FIT_TOLERANCE = 0.18
MERGE_TOLERANCE = 0.10
FIT_ITERATIONS = 3
OFFSETS = (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75)

TRAIN_BATCH = 24
MAX_EPOCHS = 30
EVAL_EVERY = 2
LEARNING_RATE = 4.0e-4

MIN_AUDIT_COVERAGE_025 = 0.95
MAX_AUDIT_AMBIGUITY_RATIO = 0.50
MIN_COSINE_LIFT = 0.010
MIN_TOP_RECALL_LIFT = 0.05
MIN_EMD_REDUCTION = 0.10
MIN_FOLD_WINS = 4
MIN_SURFACE_FEATURE_LIFT = 0.002
MAX_CLEAR_DEGRADATION = 0.002
HYPOTHESIS_JOBS = 32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def signed_circular_delta(value: np.ndarray, reference: np.ndarray | float) -> np.ndarray:
    return (
        np.asarray(value, dtype=np.float64)
        - np.asarray(reference, dtype=np.float64)
        + N_DELAY / 2.0
    ) % N_DELAY - N_DELAY / 2.0


def solve_plane(
    design: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    weighted = design * weight[:, None]
    normal = design.T @ weighted
    normal += np.diag([1e-7, 2e-3, 2e-3])
    rhs = design.T @ (weight * target)
    try:
        coefficient = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    gradient = coefficient[1:]
    norm = float(np.linalg.norm(gradient))
    if norm > MAX_GRADIENT:
        coefficient[1:] *= MAX_GRADIENT / max(norm, 1e-20)
        coefficient[0] = float(
            np.sum(weight * (target - design[:, 1:] @ coefficient[1:]))
            / max(float(np.sum(weight)), 1e-20)
        )
    return coefficient


def select_components(
    prediction: np.ndarray,
    delay: np.ndarray,
    mass: np.ndarray,
    valid: np.ndarray,
    allowance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = np.full(len(prediction), -1, dtype=np.int64)
    residual = np.full(len(prediction), np.inf, dtype=np.float64)
    chosen_mass = np.zeros(len(prediction), dtype=np.float64)
    for row in range(len(prediction)):
        active = np.flatnonzero(valid[row])
        if len(active) == 0:
            continue
        delta = signed_circular_delta(delay[row, active], prediction[row])
        index = int(np.argmin(np.abs(delta)))
        if abs(float(delta[index])) <= float(allowance[row]):
            chosen[row] = int(active[index])
            residual[row] = float(delta[index])
            chosen_mass[row] = float(mass[row, active[index]])
    return chosen, residual, chosen_mass


def fit_hypothesis(
    seed_neighbor: int,
    seed_component: int,
    query_xy: np.ndarray,
    neighbor_xy: np.ndarray,
    delay: np.ndarray,
    mass: np.ndarray,
    valid: np.ndarray,
    base_weight: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    seed_delay = float(delay[seed_neighbor, seed_component])
    seed_xy = neighbor_xy[seed_neighbor]
    distance_to_seed = np.linalg.norm(neighbor_xy - seed_xy[None, :], axis=1)
    initial_prediction = np.full(len(neighbor_xy), seed_delay, dtype=np.float64)
    allowance = INITIAL_TOLERANCE + MAX_GRADIENT * distance_to_seed
    chosen, residual, chosen_mass = select_components(
        initial_prediction, delay, mass, valid, allowance
    )
    support = chosen >= 0
    if int(np.sum(support)) < MIN_SUPPORT:
        return None

    relative_xy = neighbor_xy - query_xy[None, :]
    coefficient = np.asarray([seed_delay, 0.0, 0.0], dtype=np.float64)
    for iteration in range(FIT_ITERATIONS):
        prediction = coefficient[0] + relative_xy @ coefficient[1:]
        if iteration > 0:
            allowance = np.full(len(prediction), FIT_TOLERANCE, dtype=np.float64)
            chosen, residual, chosen_mass = select_components(
                prediction, delay, mass, valid, allowance
            )
            support = chosen >= 0
            if int(np.sum(support)) < MIN_SUPPORT:
                return None
        unwrapped = prediction + np.where(np.isfinite(residual), residual, 0.0)
        design = np.column_stack(
            [np.ones(int(np.sum(support))), relative_xy[support]]
        )
        spatial_weight = base_weight[support]
        component_weight = np.sqrt(np.maximum(chosen_mass[support], 1e-8))
        robust = np.ones(int(np.sum(support)), dtype=np.float64)
        if iteration > 0:
            robust = 1.0 / np.maximum(
                1.0, np.abs(residual[support]) / (0.5 * FIT_TOLERANCE)
            )
        weight = spatial_weight * component_weight * robust
        coefficient = solve_plane(design, unwrapped[support], weight)

    prediction = coefficient[0] + relative_xy @ coefficient[1:]
    chosen, residual, chosen_mass = select_components(
        prediction,
        delay,
        mass,
        valid,
        np.full(len(prediction), FIT_TOLERANCE, dtype=np.float64),
    )
    support = chosen >= 0
    support_count = int(np.sum(support))
    if support_count < MIN_SUPPORT:
        return None
    spatial_weight = base_weight[support]
    spatial_weight /= max(float(np.sum(spatial_weight)), 1e-20)
    rmse = float(np.sqrt(np.sum(spatial_weight * residual[support] ** 2)))
    support_weight = float(np.sum(base_weight[support]))
    support_mass = float(np.sum(base_weight[support] * chosen_mass[support]))
    mean_mass = float(np.sum(spatial_weight * chosen_mass[support]))
    mass_std = float(
        np.sqrt(np.sum(spatial_weight * (chosen_mass[support] - mean_mass) ** 2))
    )
    selected_xy = relative_xy[support]
    mean_distance = float(
        np.sum(spatial_weight * np.linalg.norm(selected_xy, axis=1))
    )
    if support_count >= 3:
        covariance = np.cov(selected_xy.T, aweights=spatial_weight)
        spatial_spread = float(np.sqrt(max(float(np.linalg.det(covariance)), 0.0)))
    else:
        spatial_spread = 0.0
    gradient = coefficient[1:]
    gradient_norm = float(np.linalg.norm(gradient))
    quality = (
        support_mass
        * support_weight
        * math.sqrt(support_count / NEIGHBORS)
        / max(rmse + 0.04, 0.04)
    )
    feature = np.asarray(
        [
            gradient[0] / MAX_GRADIENT,
            gradient[1] / MAX_GRADIENT,
            gradient_norm / MAX_GRADIENT,
            rmse / FIT_TOLERANCE,
            support_count / NEIGHBORS,
            support_weight,
            math.log(max(support_mass, 1e-10)) / 12.0,
            mean_mass,
            mass_std / max(mean_mass, 1e-8),
            mean_distance / 40.0,
            spatial_spread / 400.0,
            base_weight[seed_neighbor],
            mass[seed_neighbor, seed_component],
            quality,
        ],
        dtype=np.float32,
    )
    return float(coefficient[0] % N_DELAY), feature


@dataclass
class HypothesisSet:
    query_ids: np.ndarray
    delay: np.ndarray
    feature: np.ndarray
    prior: np.ndarray
    valid: np.ndarray
    neighbor_ids: np.ndarray
    base_weights: np.ndarray


def build_query_hypotheses(
    query_id: int,
    neighbor_ids: np.ndarray,
    base_weight: np.ndarray,
    all_xy: np.ndarray,
    component_delay: np.ndarray,
    component_weight: np.ndarray,
    component_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_weight = np.asarray(base_weight, dtype=np.float64)
    base_weight /= max(float(np.sum(base_weight)), 1e-20)
    delay = component_delay[neighbor_ids].astype(np.float64)
    mass = component_weight[neighbor_ids].astype(np.float64)
    valid = component_valid[neighbor_ids].astype(bool)
    candidates: list[tuple[float, np.ndarray]] = []
    for seed_neighbor in range(len(neighbor_ids)):
        for seed_component in np.flatnonzero(valid[seed_neighbor]):
            hypothesis = fit_hypothesis(
                seed_neighbor,
                int(seed_component),
                all_xy[query_id],
                all_xy[neighbor_ids],
                delay,
                mass,
                valid,
                base_weight,
            )
            if hypothesis is not None:
                candidates.append(hypothesis)
    candidates.sort(key=lambda item: float(item[1][-1]), reverse=True)
    retained: list[tuple[float, np.ndarray]] = []
    for candidate in candidates:
        if any(
            abs(float(signed_circular_delta(candidate[0], old[0]))) < MERGE_TOLERANCE
            for old in retained
        ):
            continue
        retained.append(candidate)
        if len(retained) == HYPOTHESES:
            break
    delay_out = np.zeros(HYPOTHESES, dtype=np.float32)
    feature_out = np.zeros((HYPOTHESES, 14), dtype=np.float32)
    prior_out = np.zeros(HYPOTHESES, dtype=np.float32)
    valid_out = np.zeros(HYPOTHESES, dtype=bool)
    if retained:
        count = len(retained)
        delay_out[:count] = [item[0] for item in retained]
        feature_out[:count] = np.stack([item[1] for item in retained])
        quality = np.asarray([max(float(item[1][-1]), 1e-12) for item in retained])
        quality /= max(float(np.sum(quality)), 1e-20)
        prior_out[:count] = quality
        valid_out[:count] = True
    return delay_out, feature_out, prior_out, valid_out


def build_hypotheses(
    data: v274.ModelSet,
    all_xy: np.ndarray,
    component: dict[str, np.ndarray],
) -> HypothesisSet:
    query_count = len(data.query_ids)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=min(HYPOTHESIS_JOBS, max(query_count, 1)), batch_size=4)(
            delayed(build_query_hypotheses)(
                int(query_id),
                data.neighbor_ids[row, :NEIGHBORS],
                data.base_weights[row, :NEIGHBORS],
                all_xy,
                component["delay"],
                component["weight"],
                component["valid"],
            )
            for row, query_id in enumerate(data.query_ids)
        )
    delay_out = np.stack([row[0] for row in rows])
    feature_out = np.stack([row[1] for row in rows])
    prior_out = np.stack([row[2] for row in rows])
    valid_out = np.stack([row[3] for row in rows])
    print(
        f"v366 hypotheses {query_count}/{query_count} "
        f"mean_active={valid_out.sum(axis=1).mean():.1f}",
        flush=True,
    )
    return HypothesisSet(
        query_ids=data.query_ids,
        delay=delay_out,
        feature=feature_out,
        prior=prior_out,
        valid=valid_out,
        neighbor_ids=data.neighbor_ids[:, :NEIGHBORS],
        base_weights=data.base_weights[:, :NEIGHBORS],
    )


def coverage_metrics(
    hypothesis: HypothesisSet,
    component: dict[str, np.ndarray],
) -> dict[str, float]:
    result: dict[str, list[float]] = {
        "raw_coverage_010": [],
        "raw_coverage_025": [],
        "raw_ambiguity_025": [],
        "surface_coverage_010": [],
        "surface_coverage_025": [],
        "surface_ambiguity_025": [],
    }
    for row, point_id in enumerate(hypothesis.query_ids):
        active = np.flatnonzero(component["valid"][point_id])
        if len(active) == 0:
            continue
        truth_delay = component["delay"][point_id, active]
        weight = component["weight"][point_id, active].astype(np.float64)
        weight /= max(float(np.sum(weight)), 1e-20)
        raw_ids = hypothesis.neighbor_ids[row]
        raw_delay = component["delay"][raw_ids][component["valid"][raw_ids]]
        surface_delay = hypothesis.delay[row, hypothesis.valid[row]]
        raw_distance = v365.circular_distance(truth_delay[:, None], raw_delay[None, :])
        surface_distance = v365.circular_distance(
            truth_delay[:, None], surface_delay[None, :]
        )
        result["raw_coverage_010"].append(float(np.sum(weight * (raw_distance.min(axis=1) <= 0.10))))
        result["raw_coverage_025"].append(float(np.sum(weight * (raw_distance.min(axis=1) <= 0.25))))
        result["raw_ambiguity_025"].append(float(np.sum(weight * np.sum(raw_distance <= 0.25, axis=1))))
        if len(surface_delay):
            result["surface_coverage_010"].append(
                float(np.sum(weight * (surface_distance.min(axis=1) <= 0.10)))
            )
            result["surface_coverage_025"].append(
                float(np.sum(weight * (surface_distance.min(axis=1) <= 0.25)))
            )
            result["surface_ambiguity_025"].append(
                float(np.sum(weight * np.sum(surface_distance <= 0.25, axis=1)))
            )
        else:
            result["surface_coverage_010"].append(0.0)
            result["surface_coverage_025"].append(0.0)
            result["surface_ambiguity_025"].append(0.0)
    return {key: float(np.mean(value)) for key, value in result.items()}


def raw_sets(
    train_ids: np.ndarray,
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    geometry: tuple,
    node_raw: np.ndarray,
) -> tuple[v274.ModelSet, v274.ModelSet, np.ndarray]:
    all_pos = geometry[0]
    node_mean = node_raw[bank_ids].mean(axis=0).astype(np.float32)
    node_std = np.maximum(node_raw[bank_ids].std(axis=0), 1e-5).astype(np.float32)
    node = np.clip((node_raw - node_mean) / node_std, -8.0, 8.0).astype(np.float32)

    def build(query: np.ndarray) -> v274.ModelSet:
        take = min(NEIGHBORS + 1, len(bank_ids))
        distance, local = cKDTree(all_pos[bank_ids, :2]).query(
            all_pos[query, :2], k=take
        )
        if take == 1:
            distance = distance[:, None]
            local = local[:, None]
        absolute = bank_ids[local]
        neighbor_ids = np.empty((len(query), NEIGHBORS), dtype=np.int64)
        neighbor_distance = np.empty((len(query), NEIGHBORS), dtype=np.float32)
        for row, query_id in enumerate(query):
            keep = absolute[row] != query_id
            candidates = absolute[row, keep]
            candidate_distance = distance[row, keep]
            if len(candidates) < NEIGHBORS:
                raise RuntimeError("insufficient non-self eikonal neighbors")
            neighbor_ids[row] = candidates[:NEIGHBORS]
            neighbor_distance[row] = candidate_distance[:NEIGHBORS]
        weight = 1.0 / np.maximum(neighbor_distance, 0.05) ** 2
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-20)
        return v274.ModelSet(
            query_ids=query.astype(np.int64),
            query_node=node[query].astype(np.float32),
            neighbor_ids=neighbor_ids,
            edge=np.zeros((len(query), NEIGHBORS, 1), dtype=np.float32),
            base_weights=weight.astype(np.float32),
        )

    return build(train_ids), build(query_ids), node


def run_audit(
    folds: list[np.ndarray],
    valid_ids: np.ndarray,
    geometry: tuple,
    node_raw: np.ndarray,
    component: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], bool]:
    all_xy = geometry[0][:, :2]
    rows: list[dict[str, object]] = []
    cache_ids: list[np.ndarray] = []
    cache_delay: list[np.ndarray] = []
    cache_valid: list[np.ndarray] = []
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        _fit, outer_set, _node = raw_sets(
            fit_ids, outer_ids, fit_ids, geometry, node_raw
        )
        hypothesis = build_hypotheses(outer_set, all_xy, component)
        metric = coverage_metrics(hypothesis, component)
        rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "hypothesis_count": summarize(hypothesis.valid.sum(axis=1)),
                "coverage": metric,
                "leakage_assertions": {"outer_in_hypothesis_donor_bank": 0},
            }
        )
        cache_ids.append(outer_ids)
        cache_delay.append(hypothesis.delay)
        cache_valid.append(hypothesis.valid)
        print(
            f"v366 audit fold={fold} coverage025={metric['surface_coverage_025']:.4f} "
            f"ambiguity={metric['surface_ambiguity_025']:.2f} "
            f"raw={metric['raw_ambiguity_025']:.2f}",
            flush=True,
        )
    mean = {
        key: float(np.mean([row["coverage"][key] for row in rows]))
        for key in rows[0]["coverage"]
    }
    ambiguity_ratio = mean["surface_ambiguity_025"] / max(mean["raw_ambiguity_025"], 1e-20)
    passed = bool(
        mean["surface_coverage_025"] >= MIN_AUDIT_COVERAGE_025
        and ambiguity_ratio <= MAX_AUDIT_AMBIGUITY_RATIO
    )
    np.savez_compressed(
        AUDIT_CACHE,
        ids=np.concatenate(cache_ids),
        delay=np.concatenate(cache_delay),
        valid=np.concatenate(cache_valid),
    )
    return rows, passed


@dataclass
class ModelData:
    query_ids: np.ndarray
    query_node: np.ndarray
    delay: np.ndarray
    feature: np.ndarray
    prior: np.ndarray
    valid: np.ndarray
    anchor: np.ndarray


def model_data(
    hypothesis: HypothesisSet,
    query_node: np.ndarray,
    truth_d: np.ndarray,
) -> ModelData:
    anchor = v365.normalize_rows(
        np.sum(
            hypothesis.base_weights[:, :, None] * truth_d[hypothesis.neighbor_ids],
            axis=1,
            dtype=np.float64,
        )
    )
    return ModelData(
        query_ids=hypothesis.query_ids,
        query_node=query_node.astype(np.float32),
        delay=hypothesis.delay,
        feature=hypothesis.feature,
        prior=hypothesis.prior,
        valid=hypothesis.valid,
        anchor=anchor,
    )


def shuffled_surface_features(data: ModelData, seed: int) -> ModelData:
    feature = np.zeros_like(data.feature)
    source = data.feature[data.valid]
    order = np.random.default_rng(seed).permutation(len(source))
    feature[data.valid] = source[order]
    return ModelData(
        query_ids=data.query_ids,
        query_node=data.query_node,
        delay=data.delay,
        feature=feature,
        prior=data.prior,
        valid=data.valid,
        anchor=data.anchor,
    )


class EikonalSurfaceModel(nn.Module):
    def __init__(self, node_dim: int, feature_dim: int):
        super().__init__()
        hidden = 128
        self.query_encoder = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, hidden)
        )
        self.surface_encoder = nn.Sequential(
            nn.Linear(feature_dim + 6, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, hidden),
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(hidden * 3, 256), nn.SiLU(), nn.LayerNorm(256), nn.Linear(256, hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=384,
            dropout=0.04,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.energy = nn.Linear(hidden, 1)
        self.shift = nn.Linear(hidden, len(OFFSETS))
        self.gate = nn.Sequential(nn.Linear(hidden * 2, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.energy.weight)
        nn.init.zeros_(self.energy.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.register_buffer("offsets", torch.tensor(OFFSETS, dtype=torch.float32))
        self.register_buffer(
            "shift_prior",
            torch.tensor([-2.0 * abs(value) for value in OFFSETS], dtype=torch.float32),
        )

    def render(
        self,
        delay: torch.Tensor,
        energy: torch.Tensor,
        shift_probability: torch.Tensor,
    ) -> torch.Tensor:
        destination = delay[:, :, None] + self.offsets
        bins = torch.arange(N_DELAY, device=delay.device, dtype=delay.dtype)
        delta = bins[None, None, None, :] - destination[:, :, :, None]
        delta = torch.remainder(delta + N_DELAY / 2.0, N_DELAY) - N_DELAY / 2.0
        atom = torch.square(torch.sinc(delta) / torch.sinc(delta / N_DELAY))
        atom /= atom.sum(dim=3, keepdim=True).clamp_min(1e-30)
        output = torch.einsum("bho,bhon->bn", energy[:, :, None] * shift_probability, atom)
        return output / output.sum(dim=1, keepdim=True).clamp_min(1e-30)

    def forward(
        self,
        query_node: torch.Tensor,
        delay: torch.Tensor,
        feature: torch.Tensor,
        prior: torch.Tensor,
        valid: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        angle = 2.0 * math.pi * delay / N_DELAY
        periodic = torch.stack(
            [
                torch.sin(angle), torch.cos(angle),
                torch.sin(2.0 * angle), torch.cos(2.0 * angle),
                torch.sin(4.0 * angle), torch.cos(4.0 * angle),
            ],
            dim=2,
        )
        q = self.query_encoder(query_node)
        h = self.surface_encoder(torch.cat([feature, periodic], dim=2))
        q_token = q[:, None, :].expand_as(h)
        token = self.token_encoder(torch.cat([q_token, h, q_token * h], dim=2))
        mask = ~valid
        token = self.transformer(token, src_key_padding_mask=mask)
        logits = torch.log(prior.clamp_min(1e-12)) + 2.5 * torch.tanh(self.energy(token)[:, :, 0])
        logits = logits.masked_fill(mask, -1e9)
        energy = torch.softmax(logits, dim=1)
        shift_probability = torch.softmax(
            self.shift_prior[None, None, :] + 2.0 * torch.tanh(self.shift(token)), dim=2
        )
        proposal = self.render(delay, energy, shift_probability)
        pooled = torch.sum(energy[:, :, None] * token, dim=1)
        gate = 0.80 * torch.sigmoid(self.gate(torch.cat([q, pooled], dim=1)))
        prediction = (1.0 - gate) * anchor + gate * proposal
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {
            "gate": gate[:, 0],
            "energy": energy,
            "energy_logits": logits,
            "shift_probability": shift_probability,
        }


def tensor_batch(data: ModelData, rows: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.asarray(getattr(data, key)[rows])).to(device)
        for key in ("query_node", "delay", "feature", "prior", "valid", "anchor")
    }


def loss_function(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(prediction, truth, dim=1).mean()
    root = 1.0 - F.cosine_similarity(
        torch.sqrt(prediction.clamp_min(1e-12)),
        torch.sqrt(truth.clamp_min(1e-12)),
        dim=1,
    ).mean()
    emd = torch.mean(torch.abs(torch.cumsum(prediction - truth, dim=1)))
    return cosine + 0.08 * root + 0.12 * emd


@torch.no_grad()
def predict(
    model: EikonalSurfaceModel,
    data: ModelData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty((len(data.query_ids), N_DELAY), dtype=np.float32)
    gate = np.empty(len(data.query_ids), dtype=np.float32)
    energy = np.empty((len(data.query_ids), HYPOTHESES), dtype=np.float32)
    for start in range(0, len(data.query_ids), TRAIN_BATCH):
        end = min(start + TRAIN_BATCH, len(data.query_ids))
        batch = tensor_batch(data, np.arange(start, end), device)
        block, diagnostics = model(**batch)
        output[start:end] = block.cpu().numpy()
        gate[start:end] = diagnostics["gate"].cpu().numpy()
        energy[start:end] = diagnostics["energy"].cpu().numpy()
    return output, gate, energy


def fit_model(
    train: ModelData,
    validation: ModelData | None,
    truth_d: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[EikonalSurfaceModel, dict[str, object]]:
    set_seed(seed)
    model = EikonalSurfaceModel(train.query_node.shape[1], train.feature.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    best_epoch = epochs
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    if validation is not None:
        best_epoch = 0
        best_score = float(np.mean(v365.row_cos(validation.anchor, truth_d[validation.query_ids])))
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train.query_ids))
        losses = []
        for start in range(0, len(order), TRAIN_BATCH):
            rows = order[start : start + TRAIN_BATCH]
            batch = tensor_batch(train, rows, device)
            target = torch.from_numpy(truth_d[train.query_ids[rows]]).to(device)
            prediction, _ = model(**batch)
            loss = loss_function(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if validation is not None and (epoch % EVAL_EVERY == 0 or epoch == epochs):
            prediction, gate, _energy = predict(model, validation, device)
            score = float(np.mean(v365.row_cos(prediction, truth_d[validation.query_ids])))
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_pdp": score,
                    "validation_gate": float(np.mean(gate)),
                }
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"v366 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_validation_pdp": (
            float(best_score) if validation is not None and np.isfinite(best_score) else None
        ),
        "history": history,
    }


def surface_ap(
    energy: np.ndarray,
    data: ModelData,
    component: dict[str, np.ndarray],
) -> float:
    labels = []
    scores = []
    for row, point_id in enumerate(data.query_ids):
        active = np.flatnonzero(component["valid"][point_id])
        if len(active) == 0:
            continue
        truth_delay = component["delay"][point_id, active]
        keep = data.valid[row]
        distance = v365.circular_distance(
            data.delay[row, keep, None], truth_delay[None, :]
        ).min(axis=1)
        labels.append((distance <= 0.25).astype(np.int8))
        scores.append(energy[row, keep])
    return float(average_precision_score(np.concatenate(labels), np.concatenate(scores)))


def run_training(
    folds: list[np.ndarray],
    valid_ids: np.ndarray,
    truth_d: np.ndarray,
    geometry: tuple,
    node_raw: np.ndarray,
    component: dict[str, np.ndarray],
    regimes: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    all_xy = geometry[0][:, :2]
    fold_rows: list[dict[str, object]] = []
    oof: dict[str, list[np.ndarray]] = {
        "ids": [], "truth": [], "baseline": [], "shuffled": [], "full": []
    }
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(folds[(fold + 1) % len(folds)], fit_ids)
        inner_fit = np.setdiff1d(fit_ids, inner_ids)
        inner_train_set, inner_val_set, inner_node = raw_sets(
            inner_fit, inner_ids, inner_fit, geometry, node_raw
        )
        outer_train_set, outer_val_set, outer_node = raw_sets(
            fit_ids, outer_ids, fit_ids, geometry, node_raw
        )
        inner_train_h = build_hypotheses(inner_train_set, all_xy, component)
        inner_val_h = build_hypotheses(inner_val_set, all_xy, component)
        outer_train_h = build_hypotheses(outer_train_set, all_xy, component)
        outer_val_h = build_hypotheses(outer_val_set, all_xy, component)
        inner_train = model_data(
            inner_train_h, inner_node[inner_train_h.query_ids], truth_d
        )
        inner_val = model_data(
            inner_val_h, inner_node[inner_val_h.query_ids], truth_d
        )
        outer_train = model_data(
            outer_train_h, outer_node[outer_train_h.query_ids], truth_d
        )
        outer_val = model_data(
            outer_val_h, outer_node[outer_val_h.query_ids], truth_d
        )
        modes = {
            "full": (inner_train, inner_val, outer_train, outer_val),
            "shuffled": (
                shuffled_surface_features(inner_train, 366_000 + fold),
                shuffled_surface_features(inner_val, 367_000 + fold),
                shuffled_surface_features(outer_train, 368_000 + fold),
                shuffled_surface_features(outer_val, 369_000 + fold),
            ),
        }
        predictions: dict[str, np.ndarray] = {}
        energies: dict[str, np.ndarray] = {}
        training_report: dict[str, object] = {}
        for mode_index, (mode, data) in enumerate(modes.items()):
            inner_train_mode, inner_val_mode, outer_train_mode, outer_val_mode = data
            inner_model, inner_report = fit_model(
                inner_train_mode, inner_val_mode, truth_d,
                3660 + 10 * fold + mode_index, MAX_EPOCHS, device,
            )
            selected_epoch = int(inner_report["best_epoch"])
            del inner_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if selected_epoch == 0:
                prediction = outer_val_mode.anchor.copy()
                gate = np.zeros(len(outer_ids), dtype=np.float32)
                energy = outer_val_mode.prior.copy()
                outer_report: dict[str, object] = {"best_epoch": 0}
            else:
                model, outer_report = fit_model(
                    outer_train_mode, None, truth_d,
                    4660 + 10 * fold + mode_index, selected_epoch, device,
                )
                prediction, gate, energy = predict(model, outer_val_mode, device)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            predictions[mode] = prediction
            energies[mode] = energy
            training_report[mode] = {
                "selected_epoch": selected_epoch,
                "inner": inner_report,
                "outer": outer_report,
                "gate": summarize(gate),
            }

        truth = truth_d[outer_ids]
        base_metric = v365.evaluate(outer_val.anchor, truth, outer_ids, component)
        shuffled_metric = v365.evaluate(
            predictions["shuffled"], truth, outer_ids, component
        )
        full_metric = v365.evaluate(predictions["full"], truth, outer_ids, component)
        regime_rows = {}
        for regime_name in ("clear", "light", "medium", "heavy"):
            mask = regimes[outer_ids] == regime_name
            if np.any(mask):
                regime_rows[regime_name] = {
                    "count": int(np.sum(mask)),
                    "cosine_lift": float(np.mean(
                        full_metric["cosine_rows"][mask] - base_metric["cosine_rows"][mask]
                    )),
                }
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "baseline": v365.metric_summary(base_metric),
                "shuffled_surface": v365.metric_summary(shuffled_metric),
                "full": v365.metric_summary(full_metric),
                "full_delta": {
                    "cosine": float(full_metric["cosine"] - base_metric["cosine"]),
                    "top_path_mass_recall": float(
                        full_metric["top_path_mass_recall"] - base_metric["top_path_mass_recall"]
                    ),
                    "delay_emd": float(full_metric["delay_emd"] - base_metric["delay_emd"]),
                },
                "full_vs_shuffled_cosine": float(
                    full_metric["cosine"] - shuffled_metric["cosine"]
                ),
                "path_existence_ap": {
                    "prior": surface_ap(outer_val.prior, outer_val, component),
                    "shuffled": surface_ap(energies["shuffled"], outer_val, component),
                    "full": surface_ap(energies["full"], outer_val, component),
                },
                "regimes": regime_rows,
                "training": training_report,
                "leakage_assertions": {"outer_in_donor_bank": 0},
            }
        )
        oof["ids"].append(outer_ids)
        oof["truth"].append(truth)
        oof["baseline"].append(outer_val.anchor)
        oof["shuffled"].append(predictions["shuffled"])
        oof["full"].append(predictions["full"])
        print(
            f"v366 fold={fold} base={base_metric['cosine']:.6f} "
            f"shuffle={shuffled_metric['cosine']:.6f} full={full_metric['cosine']:.6f}",
            flush=True,
        )
    merged = {key: np.concatenate(value) for key, value in oof.items()}
    order = np.argsort(merged["ids"])
    for key in merged:
        merged[key] = merged[key][order]
    np.savez_compressed(OOF_OUT, **merged)
    return fold_rows, merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    for path in (
        TARGET_CACHE, v297.COMPONENT_CACHE, v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE, v274.SIGNATURE_CACHE, v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    target = np.load(TARGET_CACHE)
    truth_d = v365.normalize_rows(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    component_cache = np.load(v297.COMPONENT_CACHE)
    component = {key: component_cache[key] for key in component_cache.files}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    geometry = v274.load_geometry()
    all_pos, signatures, blocker_count, scene_features = geometry[:4]
    node_raw = v274.build_node_features(all_pos, signatures, blocker_count, scene_features)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    audit_rows, audit_passed = run_audit(
        folds, valid_ids, geometry, node_raw, component
    )
    audit_mean = {
        key: float(np.mean([row["coverage"][key] for row in audit_rows]))
        for key in audit_rows[0]["coverage"]
    }
    ambiguity_ratio = audit_mean["surface_ambiguity_025"] / max(
        audit_mean["raw_ambiguity_025"], 1e-20
    )
    fold_rows = None
    summary = None
    decision = {
        "audit_coverage_025": audit_mean["surface_coverage_025"],
        "audit_ambiguity_ratio": ambiguity_ratio,
        "audit_passed": audit_passed,
        "module_passed": False,
    }
    if audit_passed and not args.audit_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        propagation_geometry = np.load(v238.GEOM_CACHE)
        regimes = regime_of(propagation_geometry["train_blocked"].astype(np.float32))
        fold_rows, merged = run_training(
            folds, valid_ids, truth_d, geometry, node_raw, component, regimes, device
        )
        base_metric = v365.evaluate(
            merged["baseline"], merged["truth"], merged["ids"], component
        )
        shuffled_metric = v365.evaluate(
            merged["shuffled"], merged["truth"], merged["ids"], component
        )
        full_metric = v365.evaluate(
            merged["full"], merged["truth"], merged["ids"], component
        )
        cosine_lift = float(full_metric["cosine"] - base_metric["cosine"])
        recall_lift = float(
            full_metric["top_path_mass_recall"] - base_metric["top_path_mass_recall"]
        )
        emd_reduction = float(
            (base_metric["delay_emd"] - full_metric["delay_emd"])
            / max(float(base_metric["delay_emd"]), 1e-20)
        )
        fold_wins = int(sum(row["full_delta"]["cosine"] > 0 for row in fold_rows))
        full_vs_shuffled = float(full_metric["cosine"] - shuffled_metric["cosine"])
        regime_summary = {}
        for regime_name in ("clear", "light", "medium", "heavy"):
            mask = regimes[merged["ids"]] == regime_name
            regime_summary[regime_name] = {
                "count": int(np.sum(mask)),
                "cosine_lift": float(np.mean(
                    full_metric["cosine_rows"][mask] - base_metric["cosine_rows"][mask]
                )),
            }
        module_passed = bool(
            cosine_lift >= MIN_COSINE_LIFT
            and recall_lift >= MIN_TOP_RECALL_LIFT
            and emd_reduction >= MIN_EMD_REDUCTION
            and fold_wins >= MIN_FOLD_WINS
            and full_vs_shuffled >= MIN_SURFACE_FEATURE_LIFT
            and regime_summary["medium"]["cosine_lift"] > 0.0
            and regime_summary["heavy"]["cosine_lift"] > 0.0
            and regime_summary["clear"]["cosine_lift"] >= -MAX_CLEAR_DEGRADATION
        )
        summary = {
            "baseline": v365.metric_summary(base_metric),
            "shuffled_surface": v365.metric_summary(shuffled_metric),
            "full": v365.metric_summary(full_metric),
            "cosine_lift": cosine_lift,
            "top_path_mass_recall_lift": recall_lift,
            "delay_emd_reduction_fraction": emd_reduction,
            "fold_wins": fold_wins,
            "full_vs_shuffled_cosine": full_vs_shuffled,
            "regimes": regime_summary,
        }
        decision.update(
            {
                "cosine_lift_required": MIN_COSINE_LIFT,
                "top_path_mass_recall_lift_required": MIN_TOP_RECALL_LIFT,
                "delay_emd_reduction_required": MIN_EMD_REDUCTION,
                "fold_wins_required": MIN_FOLD_WINS,
                "surface_feature_lift_required": MIN_SURFACE_FEATURE_LIFT,
                "module_passed": module_passed,
            }
        )

    report = {
        "method": VERSION,
        "status": (
            "strict_pdp_module_passed"
            if decision["module_passed"]
            else "audit_passed_training_not_run"
            if audit_passed and args.audit_only
            else "analysis_only_no_candidate"
        ),
        "scope": "normalized global D only; no full-H, phase, total power, Sionna target, or ZIP",
        "primary_hypothesis": (
            "A physical path forms a locally affine delay surface with gradient bounded by "
            "the eikonal path-length law. Query-local surface fitting should retain true "
            "candidate coverage while removing same-delay donor ambiguity."
        ),
        "compliance": {
            "uses": [
                "official train H continuous components and normalized PDP",
                "official train/test planar coordinates and point-cloud features",
                "theoretical delay-bin per meter conversion",
            ],
            "does_not_use": [
                "global source identity, restored mesh/material, or RT channel coefficients",
                "Sionna/synthetic labels or v212 teacher",
                "outer H before final scoring",
                "full-H projection or submission packaging",
            ],
        },
        "parameters": {
            "theoretical_gradient_bin_per_m": THEORETICAL_GRADIENT,
            "maximum_gradient_bin_per_m": MAX_GRADIENT,
            "fit_tolerance_bin": FIT_TOLERANCE,
            "neighbors": NEIGHBORS,
            "components_per_neighbor": COMPONENT_COUNT,
            "maximum_hypotheses": HYPOTHESES,
            "minimum_neighbor_support": MIN_SUPPORT,
            "hypothesis_parallel_jobs": HYPOTHESIS_JOBS,
            "folds": "v315 five mutually exclusive isolation-stratified spatial folds",
            "control": "identical network with physical surface features shuffled",
        },
        "audit_rows": audit_rows,
        "audit_summary": {
            **audit_mean,
            "surface_to_raw_ambiguity_ratio": ambiguity_ratio,
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision": decision,
        "outputs": {
            "audit_cache": PATHS.relative(AUDIT_CACHE),
            "oof": PATHS.relative(OOF_OUT) if OOF_OUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["audit_summary"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
