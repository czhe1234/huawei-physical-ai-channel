#!/usr/bin/env python3
"""Learn a strict official-only router over whole-spectrum mode templates."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 406
VERSION = "v406_neural_propagation_mode_router"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v406_neural_propagation_mode_router_oof.npz"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

ANCHOR_K = 16
DONOR_K = 32
ALPHAS = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float32)
PCA_DIM = 24
DISTANCE_FLOOR = 0.25
MAX_EPOCHS = 80
EVAL_EVERY = 5
PATIENCE_EVALS = 4
BATCH_SIZE = 96
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 3e-4
SOFT_TARGET_TEMPERATURE = 0.02

MIN_ROUTER_DELTA = 0.01
MIN_FOLD_WINS = 4
MIN_CONTROL_MARGIN = 0.002


@dataclass
class RouterData:
    query_ids: np.ndarray
    donor_ids: np.ndarray
    anchor: np.ndarray
    scores: np.ndarray
    q_geometry: np.ndarray
    donor_geometry: np.ndarray
    relation_geometry: np.ndarray
    q_full: np.ndarray
    donor_full: np.ndarray
    relation_full: np.ndarray


class ModeRouter(nn.Module):
    def __init__(self, q_dim: int, donor_dim: int, relation_dim: int) -> None:
        super().__init__()
        hidden = 128
        self.query_encoder = nn.Sequential(
            nn.Linear(q_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.donor_encoder = nn.Sequential(
            nn.Linear(donor_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.08),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=384,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.competition = nn.TransformerEncoder(layer, num_layers=2)
        self.alpha_encoder = nn.Sequential(
            nn.Linear(3, 48), nn.GELU(), nn.Linear(48, hidden)
        )
        self.option_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.no_action_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        alpha = torch.from_numpy(
            np.stack([ALPHAS, ALPHAS**2, 1.0 - ALPHAS], axis=1)
        )
        self.register_buffer("alpha_features", alpha)

    def forward(
        self, q: torch.Tensor, donor: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        q_hidden = self.query_encoder(q)
        donor_hidden = self.donor_encoder(donor)
        relation_hidden = self.relation_encoder(relation)
        q_expanded = q_hidden[:, None, :].expand_as(donor_hidden)
        token = self.fusion(
            torch.cat([q_expanded, donor_hidden, relation_hidden], dim=-1)
        )
        token = self.competition(token)
        alpha_hidden = self.alpha_encoder(self.alpha_features)
        option = token[:, :, None, :] + alpha_hidden[None, None, :, :]
        option_logit = self.option_head(option).squeeze(-1).flatten(1)
        no_action = self.no_action_head(
            torch.cat([q_hidden, token.mean(dim=1), token.amax(dim=1)], dim=-1)
        )
        return torch.cat([no_action, option_logit], dim=1)


def normalize(x: np.ndarray) -> np.ndarray:
    return v402.normalize(np.asarray(x, dtype=np.float32))


def row_cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def bank_cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), 1, -1)
    y = np.asarray(right, dtype=np.float64).reshape(*right.shape[:2], -1)
    return (
        np.sum(x * y, axis=-1)
        / np.maximum(
            np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1), 1e-30
        )
    ).astype(np.float32)


def paired_bank_cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64).reshape(*left.shape[:2], -1)
    y = np.asarray(right, dtype=np.float64).reshape(*right.shape[:2], -1)
    return (
        np.sum(x * y, axis=-1)
        / np.maximum(
            np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1), 1e-30
        )
    ).astype(np.float32)


def find_neighbors(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    xy: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(xy[bank_ids])
    raw_count = min(count + 1, len(bank_ids))
    distance, local = tree.query(xy[query_ids], k=raw_count)
    if raw_count == 1:
        distance = distance[:, None]
        local = local[:, None]
    mapped = bank_ids[local]
    output_ids = np.empty((len(query_ids), count), dtype=np.int64)
    output_distance = np.empty((len(query_ids), count), dtype=np.float64)
    bank_membership = np.zeros(len(xy), dtype=bool)
    bank_membership[bank_ids] = True
    for row, query_id in enumerate(query_ids):
        keep = mapped[row] != query_id if bank_membership[query_id] else np.ones(raw_count, bool)
        selected_ids = mapped[row, keep][:count]
        selected_distance = np.asarray(distance[row])[keep][:count]
        if len(selected_ids) != count:
            raise AssertionError("neighbor bank is too small after self exclusion")
        output_ids[row] = selected_ids
        output_distance[row] = selected_distance
    return output_ids, output_distance


def weighted_anchor(
    target: np.ndarray, neighbor_ids: np.ndarray, distance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weight = 1.0 / np.maximum(distance[:, :ANCHOR_K], DISTANCE_FLOOR) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    anchor = np.einsum(
        "qk,qksh->qsh",
        weight,
        target[neighbor_ids[:, :ANCHOR_K]],
        optimize=True,
    )
    return normalize(anchor), weight.astype(np.float32)


def mixture_cosines(
    anchor: np.ndarray,
    donor: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    a = np.asarray(anchor, dtype=np.float64).reshape(len(anchor), 1, -1)
    d = np.asarray(donor, dtype=np.float64).reshape(*donor.shape[:2], -1)
    t = np.asarray(truth, dtype=np.float64).reshape(len(truth), 1, -1)
    at = np.sum(a * t, axis=-1)
    dt = np.sum(d * t, axis=-1)
    aa = np.sum(a * a, axis=-1)
    dd = np.sum(d * d, axis=-1)
    ad = np.sum(a * d, axis=-1)
    tt = np.sum(t * t, axis=-1)
    alpha = ALPHAS.astype(np.float64)[None, None, :]
    one_minus = 1.0 - alpha
    numerator = one_minus * at[:, :, None] + alpha * dt[:, :, None]
    norm2 = (
        one_minus**2 * aa[:, :, None]
        + alpha**2 * dd[:, :, None]
        + 2.0 * one_minus * alpha * ad[:, :, None]
    )
    return (
        numerator / np.maximum(np.sqrt(norm2 * tt[:, :, None]), 1e-30)
    ).astype(np.float32)


def option_scores(
    anchor: np.ndarray, donor: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    sh = mixture_cosines(anchor, donor, truth)
    pdp = mixture_cosines(
        anchor.sum(axis=2), donor.sum(axis=3), truth.sum(axis=2)
    )
    baseline = 0.5 * (
        row_cos(anchor, truth)
        + row_cos(anchor.sum(axis=2), truth.sum(axis=2))
    )
    return np.concatenate(
        [baseline[:, None], (0.5 * (sh + pdp)).reshape(len(anchor), -1)], axis=1
    ).astype(np.float32)


def build_router_data(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    xy: np.ndarray,
    node: np.ndarray,
    target: np.ndarray,
    log_power: np.ndarray,
    descriptor: np.ndarray,
    target_latent: np.ndarray,
    pca: PCA,
) -> RouterData:
    fit_neighbor_ids, fit_distance = find_neighbors(fit_ids, fit_ids, xy, DONOR_K)
    fit_anchor, fit_anchor_weight = weighted_anchor(target, fit_neighbor_ids, fit_distance)
    query_neighbor_ids, query_distance = find_neighbors(query_ids, fit_ids, xy, DONOR_K)
    query_anchor, query_anchor_weight = weighted_anchor(
        target, query_neighbor_ids, query_distance
    )
    lookup = np.full(len(target), -1, dtype=np.int64)
    lookup[fit_ids] = np.arange(len(fit_ids))
    donor_ids = query_neighbor_ids
    donor = target[donor_ids]
    donor_anchor = fit_anchor[lookup[donor_ids]]
    query_descriptor = v257.spectrum_descriptor(
        query_anchor, query_anchor.sum(axis=2)
    )
    fit_anchor_descriptor = v257.spectrum_descriptor(
        fit_anchor, fit_anchor.sum(axis=2)
    )
    donor_descriptor = descriptor[donor_ids]
    donor_anchor_descriptor = fit_anchor_descriptor[lookup[donor_ids]]
    query_latent = pca.transform(
        np.sqrt(np.maximum(query_anchor, 0.0)).reshape(len(query_anchor), -1)
    ).astype(np.float32)
    fit_anchor_latent = pca.transform(
        np.sqrt(np.maximum(fit_anchor, 0.0)).reshape(len(fit_anchor), -1)
    ).astype(np.float32)
    donor_latent = target_latent[donor_ids]
    donor_anchor_latent = fit_anchor_latent[lookup[donor_ids]]
    query_power = np.sum(
        log_power[query_neighbor_ids[:, :ANCHOR_K]] * query_anchor_weight, axis=1
    ).astype(np.float32)
    fit_anchor_power = np.sum(
        log_power[fit_neighbor_ids[:, :ANCHOR_K]] * fit_anchor_weight, axis=1
    ).astype(np.float32)
    donor_power = log_power[donor_ids].astype(np.float32)
    donor_anchor_power = fit_anchor_power[lookup[donor_ids]]

    query_node = node[query_ids].astype(np.float32)
    donor_node = node[donor_ids].astype(np.float32)
    q_geometry = query_node
    donor_geometry = donor_node
    delta_node = query_node[:, None, :] - donor_node
    delta_xy = xy[donor_ids] - xy[query_ids, None, :]
    bearing = np.arctan2(delta_xy[..., 1], delta_xy[..., 0])
    rank = np.broadcast_to(
        np.arange(DONOR_K, dtype=np.float32)[None, :] / (DONOR_K - 1),
        query_distance.shape,
    )
    basic_relation = np.concatenate(
        [
            delta_node,
            np.abs(delta_node),
            delta_xy.astype(np.float32),
            query_distance[..., None].astype(np.float32),
            np.log1p(query_distance)[..., None].astype(np.float32),
            np.sin(bearing)[..., None].astype(np.float32),
            np.cos(bearing)[..., None].astype(np.float32),
            rank[..., None],
        ],
        axis=-1,
    )
    relation_geometry = basic_relation

    q_full = np.concatenate(
        [query_node, query_descriptor, query_latent, query_power[:, None]], axis=1
    ).astype(np.float32)
    donor_full = np.concatenate(
        [
            donor_node,
            donor_descriptor,
            donor_anchor_descriptor,
            donor_descriptor - donor_anchor_descriptor,
            donor_latent,
            donor_anchor_latent,
            donor_latent - donor_anchor_latent,
            donor_power[..., None],
            donor_anchor_power[..., None],
            (donor_power - donor_anchor_power)[..., None],
        ],
        axis=-1,
    ).astype(np.float32)
    query_descriptor_expanded = query_descriptor[:, None, :]
    query_latent_expanded = query_latent[:, None, :]
    query_power_expanded = query_power[:, None]
    anchor_pdp = query_anchor.sum(axis=2)
    donor_pdp = donor.sum(axis=3)
    donor_anchor_pdp = donor_anchor.sum(axis=3)
    similarity = np.stack(
        [
            bank_cos(query_anchor, donor),
            bank_cos(anchor_pdp, donor_pdp),
            paired_bank_cos(donor, donor_anchor),
            paired_bank_cos(donor_pdp, donor_anchor_pdp),
            bank_cos(query_anchor, donor_anchor),
            bank_cos(anchor_pdp, donor_anchor_pdp),
        ],
        axis=-1,
    )
    relation_full = np.concatenate(
        [
            basic_relation,
            query_descriptor_expanded - donor_descriptor,
            query_descriptor_expanded - donor_anchor_descriptor,
            query_latent_expanded - donor_latent,
            query_latent_expanded - donor_anchor_latent,
            (query_power_expanded - donor_power)[..., None],
            (query_power_expanded - donor_anchor_power)[..., None],
            similarity,
        ],
        axis=-1,
    ).astype(np.float32)
    scores = option_scores(query_anchor, donor, target[query_ids])
    arrays = (
        q_geometry,
        donor_geometry,
        relation_geometry,
        q_full,
        donor_full,
        relation_full,
        scores,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise AssertionError("router features contain non-finite values")
    return RouterData(
        query_ids=query_ids,
        donor_ids=donor_ids,
        anchor=query_anchor,
        scores=scores,
        q_geometry=q_geometry,
        donor_geometry=donor_geometry,
        relation_geometry=relation_geometry,
        q_full=q_full,
        donor_full=donor_full,
        relation_full=relation_full,
    )


def feature_arrays(data: RouterData, mode: str) -> tuple[np.ndarray, ...]:
    if mode == "geometry":
        return data.q_geometry, data.donor_geometry, data.relation_geometry
    if mode == "full":
        return data.q_full, data.donor_full, data.relation_full
    raise ValueError(mode)


def standardize_features(
    data: RouterData, fit_rows: np.ndarray, mode: str
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    arrays = feature_arrays(data, mode)
    normalized: list[np.ndarray] = []
    statistics: list[np.ndarray] = []
    for array in arrays:
        fit = array[fit_rows].reshape(-1, array.shape[-1]).astype(np.float64)
        mean = fit.mean(axis=0).astype(np.float32)
        std = fit.std(axis=0).astype(np.float32)
        std[std < 1e-5] = 1.0
        normalized.append(((array - mean) / std).astype(np.float32))
        statistics.extend([mean, std])
    return tuple(normalized), tuple(statistics)


def apply_statistics(
    data: RouterData, mode: str, statistics: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, ...]:
    arrays = feature_arrays(data, mode)
    return tuple(
        ((array - statistics[2 * index]) / statistics[2 * index + 1]).astype(
            np.float32
        )
        for index, array in enumerate(arrays)
    )


def router_loss(logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    soft_target = torch.softmax(
        (scores - scores.amax(dim=1, keepdim=True)) / SOFT_TARGET_TEMPERATURE,
        dim=1,
    )
    soft_loss = torch.sum(-soft_target * F.log_softmax(logits, dim=1), dim=1).mean()
    hard_loss = F.cross_entropy(logits, scores.argmax(dim=1))
    return 0.70 * soft_loss + 0.30 * hard_loss


def train_router(
    features: tuple[np.ndarray, ...],
    scores: np.ndarray,
    train_rows: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
    validation_rows: np.ndarray | None = None,
    shuffled_labels: bool = False,
) -> tuple[ModeRouter, int, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q, donor, relation = [torch.from_numpy(value).to(device) for value in features]
    truth_score = torch.from_numpy(scores).to(device)
    train_score = scores.copy()
    if shuffled_labels:
        for row in train_rows:
            train_score[row] = train_score[row, rng.permutation(train_score.shape[1])]
    label = torch.from_numpy(train_score).to(device)
    model = ModeRouter(q.shape[-1], donor.shape[-1], relation.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = epochs
    best_validation = -np.inf
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_rows)
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            rows = torch.from_numpy(order[start : start + BATCH_SIZE]).to(device)
            logits = model(q[rows], donor[rows], relation[rows])
            loss = router_loss(logits, label[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(rows)
        if validation_rows is not None and (
            epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs
        ):
            model.eval()
            rows = torch.from_numpy(validation_rows).to(device)
            with torch.no_grad():
                choice = model(q[rows], donor[rows], relation[rows]).argmax(dim=1)
                selected = truth_score[rows].gather(1, choice[:, None])[:, 0]
                baseline = truth_score[rows, 0]
                validation_delta = float((selected - baseline).mean().cpu())
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / len(train_rows),
                    "validation_delta": validation_delta,
                }
            )
            if validation_delta > best_validation + 1e-5:
                best_validation = validation_delta
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if epoch >= 20 and stale >= PATIENCE_EVALS:
                break
    if validation_rows is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


@torch.no_grad()
def predict_choices(
    model: ModeRouter,
    features: tuple[np.ndarray, ...],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(features[0]), BATCH_SIZE):
        batch = [
            torch.from_numpy(value[start : start + BATCH_SIZE]).to(device)
            for value in features
        ]
        output.append(model(*batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.int64)


def render_choices(
    data: RouterData, choices: np.ndarray, target: np.ndarray
) -> np.ndarray:
    prediction = data.anchor.copy()
    active = choices > 0
    option = choices[active] - 1
    rank = option // len(ALPHAS)
    alpha_id = option % len(ALPHAS)
    rows = np.flatnonzero(active)
    donor = target[data.donor_ids[rows, rank]]
    alpha = ALPHAS[alpha_id].reshape(-1, 1, 1)
    prediction[active] = (1.0 - alpha) * prediction[active] + alpha * donor
    return normalize(prediction)


def evaluate_prediction(
    prediction: np.ndarray,
    truth: np.ndarray,
    baseline_values: dict[str, np.ndarray],
) -> tuple[dict[str, object], np.ndarray]:
    values = v402.point_metrics(prediction, truth)
    delta = values["weighted"] - baseline_values["weighted"]
    return (
        {
            "metrics": v402.metric_summary(prediction, truth),
            "delta": {
                key: float(np.mean(values[key] - baseline_values[key]))
                for key in values
            },
            "point_gain": summarize(delta),
            "point_win_fraction": float(np.mean(delta > 0.0)),
        },
        delta,
    )


def fold_experiment(
    fold: int,
    folds: list[np.ndarray],
    valid_ids: np.ndarray,
    xy: np.ndarray,
    node: np.ndarray,
    target: np.ndarray,
    log_power: np.ndarray,
    descriptor: np.ndarray,
    blocked: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    query_ids = folds[fold]
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=4060 + fold)
    pca.fit(np.sqrt(np.maximum(target[fit_ids], 0.0)).reshape(len(fit_ids), -1))
    target_latent = np.zeros((len(target), PCA_DIM), dtype=np.float32)
    target_latent[fit_ids] = pca.transform(
        np.sqrt(np.maximum(target[fit_ids], 0.0)).reshape(len(fit_ids), -1)
    ).astype(np.float32)
    pseudo = build_router_data(
        fit_ids,
        fit_ids,
        xy,
        node,
        target,
        log_power,
        descriptor,
        target_latent,
        pca,
    )
    outer = build_router_data(
        query_ids,
        fit_ids,
        xy,
        node,
        target,
        log_power,
        descriptor,
        target_latent,
        pca,
    )
    validation_ids = folds[(fold + 1) % len(folds)]
    validation_rows = np.flatnonzero(np.isin(fit_ids, validation_ids)).astype(np.int64)
    train_rows = np.setdiff1d(np.arange(len(fit_ids)), validation_rows)
    all_rows = np.arange(len(fit_ids), dtype=np.int64)

    fold_predictions: dict[str, np.ndarray] = {}
    training_report: dict[str, object] = {}
    for mode in ("geometry", "full"):
        inner_features, _ = standardize_features(pseudo, train_rows, mode)
        _, best_epoch, history = train_router(
            inner_features,
            pseudo.scores,
            train_rows,
            MAX_EPOCHS,
            40600 + 100 * fold + (0 if mode == "geometry" else 10),
            device,
            validation_rows=validation_rows,
        )
        final_features, statistics = standardize_features(pseudo, all_rows, mode)
        model, _, final_history = train_router(
            final_features,
            pseudo.scores,
            all_rows,
            best_epoch,
            40601 + 100 * fold + (0 if mode == "geometry" else 10),
            device,
        )
        outer_features = apply_statistics(outer, mode, statistics)
        choice = predict_choices(model, outer_features, device)
        fold_predictions[mode] = render_choices(outer, choice, target)
        training_report[mode] = {
            "selected_epoch": int(best_epoch),
            "inner_history": history,
            "final_last_loss": (
                final_history[-1]["loss"] if final_history else None
            ),
            "no_action_fraction": float(np.mean(choice == 0)),
            "selected_rank": summarize((np.maximum(choice - 1, 0) // len(ALPHAS))[choice > 0]),
            "selected_alpha": summarize(
                ALPHAS[(np.maximum(choice - 1, 0) % len(ALPHAS))[choice > 0]]
            ),
        }
        if mode == "full":
            shuffled_model, _, _ = train_router(
                final_features,
                pseudo.scores,
                all_rows,
                best_epoch,
                40602 + 100 * fold,
                device,
                shuffled_labels=True,
            )
            shuffled_choice = predict_choices(shuffled_model, outer_features, device)
            fold_predictions["shuffled"] = render_choices(
                outer, shuffled_choice, target
            )
            training_report["shuffled"] = {
                "epoch": int(best_epoch),
                "no_action_fraction": float(np.mean(shuffled_choice == 0)),
            }

    fixed_alpha = int(
        np.argmax(
            pseudo.scores[:, 1 : 1 + len(ALPHAS)].mean(axis=0)
        )
    )
    nearest_choice = np.full(len(query_ids), 1 + fixed_alpha, dtype=np.int64)
    fold_predictions["nearest"] = render_choices(outer, nearest_choice, target)
    oracle_choice = outer.scores.argmax(axis=1)
    fold_predictions["oracle"] = render_choices(outer, oracle_choice, target)

    truth = target[query_ids]
    baseline_values = v402.point_metrics(outer.anchor, truth)
    methods: dict[str, object] = {}
    point_delta: dict[str, np.ndarray] = {}
    regimes = regime_of(blocked[query_ids]).astype(str)
    for name in ("nearest", "geometry", "full", "shuffled", "oracle"):
        method, delta = evaluate_prediction(
            fold_predictions[name], truth, baseline_values
        )
        method["by_regime"] = {
            regime: float(np.mean(delta[regimes == regime]))
            for regime in ("clear", "light", "medium", "heavy")
            if np.any(regimes == regime)
        }
        methods[name] = method
        point_delta[name] = delta
    row = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "baseline": v402.metric_summary(outer.anchor, truth),
        "fixed_nearest_alpha": float(ALPHAS[fixed_alpha]),
        "methods": methods,
        "training": training_report,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "outer_truth_in_features_or_training": False,
            "outer_truth_used_for_scoring_and_oracle_only": True,
        },
    }
    arrays = {
        "ids": query_ids,
        "baseline": outer.anchor,
        "truth": truth,
        **fold_predictions,
        **{f"delta_{name}": value for name, value in point_delta.items()},
    }
    return row, arrays


def main() -> None:
    for path in (TARGET_CACHE, GEOMETRY_CACHE, v297.COMPONENT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(TARGET_CACHE)
    target = normalize(targets["sh"].astype(np.float32))
    log_power = targets["log_power"].astype(np.float32)
    descriptor = v257.spectrum_descriptor(target, target.sum(axis=2))
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    xy = positions[:, :2]
    test_xy = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    node, _ = v257.build_node_features(xy, test_xy)
    blocked = np.load(GEOMETRY_CACHE)["train_blocked"].astype(np.float32)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        targets["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, xy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    parts: list[dict[str, np.ndarray]] = []
    for fold in range(len(folds)):
        row, arrays = fold_experiment(
            fold,
            folds,
            valid_ids,
            xy,
            node,
            target,
            log_power,
            descriptor,
            blocked,
            device,
        )
        rows.append(row)
        parts.append(arrays)
        print(
            f"v406 fold={fold} nearest={row['methods']['nearest']['delta']['weighted']:+.5f} "
            f"geometry={row['methods']['geometry']['delta']['weighted']:+.5f} "
            f"full={row['methods']['full']['delta']['weighted']:+.5f} "
            f"shuffle={row['methods']['shuffled']['delta']['weighted']:+.5f} "
            f"oracle={row['methods']['oracle']['delta']['weighted']:+.5f}",
            flush=True,
        )

    aggregate: dict[str, object] = {}
    for name in ("nearest", "geometry", "full", "shuffled", "oracle"):
        delta = np.asarray(
            [row["methods"][name]["delta"]["weighted"] for row in rows]
        )
        aggregate[name] = {
            "weighted_delta": summarize(delta),
            "positive_folds": int(np.sum(delta > 0.0)),
        }
    full_delta = float(aggregate["full"]["weighted_delta"]["mean"])
    control = max(
        float(aggregate[name]["weighted_delta"]["mean"])
        for name in ("nearest", "geometry", "shuffled")
    )
    passed = bool(
        full_delta >= MIN_ROUTER_DELTA
        and aggregate["full"]["positive_folds"] >= MIN_FOLD_WINS
        and full_delta - control >= MIN_CONTROL_MARGIN
    )
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != len(np.unique(ids)) or len(ids) != len(valid_ids):
        raise AssertionError("OOF IDs are not a unique cover")
    array_names = (
        "baseline",
        "truth",
        "nearest",
        "geometry",
        "full",
        "shuffled",
        "oracle",
        "delta_nearest",
        "delta_geometry",
        "delta_full",
        "delta_shuffled",
        "delta_oracle",
    )
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate(
            [
                np.full(len(part["ids"]), fold, dtype=np.int16)
                for fold, part in enumerate(parts)
            ]
        ),
        **{
            name: np.concatenate([part[name] for part in parts])
            for name in array_names
        },
    )
    report = {
        "method": VERSION,
        "status": (
            "official_oof_passed_ready_for_carrier_deployment"
            if passed
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Official geometry and fit-only anchor state can route an unseen UE to "
            "the correct whole-spectrum propagation-mode template."
        ),
        "compliance": {
            "uses": [
                "official train H spectra",
                "official train coordinates",
                "official train/test coordinates",
                "official Round1_Map.ply-derived node features",
            ],
            "does_not_use": [
                "test H or hidden truth",
                "v212/Sionna/RT/synthetic targets or coefficients",
                "online score labels",
            ],
            "outer_truth": "used only for OOF scoring and the reported oracle ceiling",
        },
        "parameters": {
            "anchor_k": ANCHOR_K,
            "donor_k": DONOR_K,
            "alphas": ALPHAS.tolist(),
            "pca_dim": PCA_DIM,
            "epochs_max": MAX_EPOCHS,
            "soft_target_temperature": SOFT_TARGET_TEMPERATURE,
            "router": "query/donor/relation encoders plus two-layer donor-set Transformer",
            "folds": "five mutually exclusive isolation-stratified folds",
        },
        "fold_rows": rows,
        "aggregate": aggregate,
        "selection": {
            "minimum_router_delta": MIN_ROUTER_DELTA,
            "minimum_positive_folds": MIN_FOLD_WINS,
            "minimum_control_margin": MIN_CONTROL_MARGIN,
            "observed_control_margin": full_delta - control,
            "passed": passed,
            "next_action": (
                "deploy the selected normalized joint target on the strict v284 full-H carrier"
                if passed
                else "do not deploy; diagnose router regret against the v405 oracle"
            ),
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
