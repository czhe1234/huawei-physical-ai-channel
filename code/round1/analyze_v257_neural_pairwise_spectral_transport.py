#!/usr/bin/env python3
"""v257: official-pair neural spectral transport.

The v255 oracle showed that small global D/S/H shifts explain a large part of
neighbour interpolation error. This experiment learns those shifts from
official train pairs. Query truth is used only to create train-pair labels; the
deployed network receives public geometry, point-cloud features, neighbour
spectrum summaries, and raw-anchor cross-correlation diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v255_global_spectral_transport_oracle as v255
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 257
VERSION = "v257_neural_pairwise_spectral_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.artifacts / "checkpoints" / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v257_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v257_neural_transport_targets.npz"
ACTIONS = OUT_DIR / "v257_neural_transport_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
GEODESIC_CACHE = PATHS.shared_features / "geodesic_support_features.npz"
HEIGHT_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

LOCAL_GATE = 0.645604
FOLD_SEEDS = v254.FOLD_SEEDS
VAL_COUNT = 320
NEIGHBORS = 16
POWER = 2.0
D_BOUND = 4
S_BOUND = 4
H_BOUND = 3
EPOCHS = 70
BATCH_SIZE = 4096
MIN_OFFICIAL_DELTA = 5e-4
THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.65, 0.8)
BLENDS = (0.25, 0.5, 0.75, 1.0)
MODES = ("d", "sh", "joint")
SOURCES = ("direct", "self_gate")


@dataclass
class PairData:
    features: np.ndarray
    neighbour_ids: np.ndarray
    weights: np.ndarray
    raw_sh: np.ndarray
    raw_d: np.ndarray
    self_d: np.ndarray
    self_s: np.ndarray
    self_h: np.ndarray
    oracle_d: np.ndarray | None
    oracle_s: np.ndarray | None
    oracle_h: np.ndarray | None


class ShiftNet(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 192),
            nn.GELU(),
        )
        self.d_head = nn.Linear(192, 2 * D_BOUND + 1)
        self.s_head = nn.Linear(192, 2 * S_BOUND + 1)
        self.h_head = nn.Linear(192, 2 * H_BOUND + 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.d_head(hidden), self.s_head(hidden), self.h_head(hidden)


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape(
        (int(np.sum(valid)),) + (1,) * (out.ndim - 1)
    )
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def metrics(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def weighted_average(bank: np.ndarray, weight: np.ndarray) -> np.ndarray:
    shape = weight.shape + (1,) * (bank.ndim - 2)
    return normalize(
        np.sum(bank * weight.reshape(shape), axis=1, dtype=np.float64).astype(np.float32)
    )


def circular_descriptor(probability: np.ndarray) -> np.ndarray:
    p = normalize(probability)
    n = p.shape[-1]
    angle = 2.0 * np.pi * np.arange(n, dtype=np.float64) / n
    moment = np.sum(p * np.exp(1j * angle), axis=-1)
    peak = np.argmax(p, axis=-1)
    peak_angle = 2.0 * np.pi * peak / n
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1) / np.log(n)
    return np.stack(
        [
            moment.real,
            moment.imag,
            np.abs(moment),
            np.sin(peak_angle),
            np.cos(peak_angle),
            np.max(p, axis=-1),
            entropy,
        ],
        axis=-1,
    ).astype(np.float32)


def spectrum_descriptor(sh: np.ndarray, d: np.ndarray) -> np.ndarray:
    flat_shape = sh.shape[:-2]
    sh_flat = sh.reshape(-1, sh.shape[-2], sh.shape[-1])
    d_flat = d.reshape(-1, d.shape[-1])
    s_marginal = normalize(sh_flat.sum(axis=-1))
    h_marginal = normalize(sh_flat.sum(axis=-2))
    descriptor = np.concatenate(
        [
            circular_descriptor(d_flat),
            circular_descriptor(s_marginal),
            circular_descriptor(h_marginal),
        ],
        axis=-1,
    )
    return descriptor.reshape(*flat_shape, descriptor.shape[-1]).astype(np.float32)


def corr_1d_stats(
    query: np.ndarray, bank: np.ndarray, bound: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_fft = np.fft.fft(query, axis=-1)
    b_fft = np.fft.fft(bank, axis=-1)
    correlation = np.fft.ifft(q_fft[:, None] * np.conj(b_fft), axis=-1).real
    norm = np.linalg.norm(query, axis=-1)[:, None] * np.linalg.norm(bank, axis=-1)
    correlation /= np.maximum(norm[:, :, None], 1e-20)
    shifts = np.arange(-bound, bound + 1, dtype=np.int16)
    values = correlation[..., shifts % query.shape[-1]]
    order = np.argsort(values, axis=-1)
    best_index = order[..., -1]
    best = np.take_along_axis(values, best_index[..., None], axis=-1)[..., 0]
    second = np.take_along_axis(values, order[..., -2, None], axis=-1)[..., 0]
    return shifts[best_index], best.astype(np.float32), (best - second).astype(np.float32), values[..., bound].astype(np.float32)


def corr_2d_stats(
    query: np.ndarray, bank: np.ndarray, s_bound: int, h_bound: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_fft = np.fft.fftn(query, axes=(-2, -1))
    b_fft = np.fft.fftn(bank, axes=(-2, -1))
    correlation = np.fft.ifftn(
        q_fft[:, None] * np.conj(b_fft), axes=(-2, -1)
    ).real
    norm = np.linalg.norm(query.reshape(len(query), -1), axis=-1)[:, None] * np.linalg.norm(
        bank.reshape(*bank.shape[:2], -1), axis=-1
    )
    correlation /= np.maximum(norm[:, :, None, None], 1e-20)
    s_shifts = np.arange(-s_bound, s_bound + 1, dtype=np.int16)
    h_shifts = np.arange(-h_bound, h_bound + 1, dtype=np.int16)
    values = correlation[
        ...,
        s_shifts[:, None] % query.shape[-2],
        h_shifts[None, :] % query.shape[-1],
    ]
    flat = values.reshape(*values.shape[:-2], -1)
    order = np.argsort(flat, axis=-1)
    best_index = order[..., -1]
    best = np.take_along_axis(flat, best_index[..., None], axis=-1)[..., 0]
    second = np.take_along_axis(flat, order[..., -2, None], axis=-1)[..., 0]
    h_count = len(h_shifts)
    zero_index = s_bound * h_count + h_bound
    return (
        s_shifts[best_index // h_count],
        h_shifts[best_index % h_count],
        best.astype(np.float32),
        (best - second).astype(np.float32),
        flat[..., zero_index].astype(np.float32),
    )


def build_node_features(train_pos: np.ndarray, test_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scene = np.load(SCENE_CACHE)
    geodesic = np.load(GEODESIC_CACHE)
    height = np.load(HEIGHT_CACHE)
    train_scene = scene["train_features"].astype(np.float32)
    test_scene = scene["test_features"].astype(np.float32)
    train_geo = geodesic["train_features"].astype(np.float32)
    test_geo = geodesic["test_features"].astype(np.float32)
    train_blocked = height["train_blocked"].astype(np.float32)[:, None]
    test_blocked = height["test_blocked"].astype(np.float32)[:, None]

    def engineered(pos: np.ndarray) -> np.ndarray:
        delta = pos - np.asarray([50.0, 0.0], dtype=np.float64)
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        return np.concatenate(
            [
                pos.astype(np.float32),
                distance.astype(np.float32),
                np.sin(bearing)[:, None].astype(np.float32),
                np.cos(bearing)[:, None].astype(np.float32),
            ],
            axis=1,
        )

    train = np.concatenate(
        [engineered(train_pos), train_scene, train_geo, train_blocked], axis=1
    )
    test = np.concatenate(
        [engineered(test_pos), test_scene, test_geo, test_blocked], axis=1
    )
    return (
        np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        np.nan_to_num(test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    )


def build_pair_data(
    query_pos: np.ndarray,
    query_node: np.ndarray,
    bank_ids: np.ndarray,
    train_pos: np.ndarray,
    train_node: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
    query_truth_sh: np.ndarray | None,
    query_truth_d: np.ndarray | None,
    query_train_ids: np.ndarray | None,
) -> PairData:
    exclude_self = query_train_ids is not None
    search_k = NEIGHBORS + 1 if exclude_self else NEIGHBORS
    distance, local = cKDTree(train_pos[bank_ids]).query(query_pos, k=search_k)
    absolute_all = bank_ids[local]
    if exclude_self:
        absolute = np.empty((len(query_pos), NEIGHBORS), dtype=np.int64)
        kept_distance = np.empty((len(query_pos), NEIGHBORS), dtype=np.float64)
        for row, query_id in enumerate(query_train_ids):
            keep = absolute_all[row] != query_id
            absolute[row] = absolute_all[row, keep][:NEIGHBORS]
            kept_distance[row] = distance[row, keep][:NEIGHBORS]
        distance = kept_distance
    else:
        absolute = absolute_all
    weight = 1.0 / np.maximum(distance, 0.05) ** POWER
    weight /= weight.sum(axis=1, keepdims=True)
    neighbour_sh = sh[absolute]
    neighbour_d = d[absolute]
    raw_sh = weighted_average(neighbour_sh, weight)
    raw_d = weighted_average(neighbour_d, weight)

    self_d, d_peak, d_margin, d_zero = corr_1d_stats(raw_d, neighbour_d, D_BOUND)
    self_s, self_h, sh_peak, sh_margin, sh_zero = corr_2d_stats(
        raw_sh, neighbour_sh, S_BOUND, H_BOUND
    )
    raw_descriptor = spectrum_descriptor(raw_sh, raw_d)
    neighbour_descriptor = spectrum_descriptor(neighbour_sh, neighbour_d)
    q_node = np.repeat(query_node[:, None, :], NEIGHBORS, axis=1)
    n_node = train_node[absolute]
    q_desc = np.repeat(raw_descriptor[:, None, :], NEIGHBORS, axis=1)
    delta_pos = train_pos[absolute] - query_pos[:, None, :]
    basic = np.concatenate(
        [
            delta_pos.astype(np.float32),
            distance[:, :, None].astype(np.float32),
            weight[:, :, None].astype(np.float32),
        ],
        axis=-1,
    )
    correlation = np.stack(
        [
            self_d / max(D_BOUND, 1),
            d_peak,
            d_margin,
            d_zero,
            self_s / max(S_BOUND, 1),
            self_h / max(H_BOUND, 1),
            sh_peak,
            sh_margin,
            sh_zero,
        ],
        axis=-1,
    ).astype(np.float32)
    features = np.concatenate(
        [
            q_node,
            n_node,
            q_node - n_node,
            q_desc,
            neighbour_descriptor,
            q_desc - neighbour_descriptor,
            basic,
            correlation,
        ],
        axis=-1,
    ).reshape(len(query_pos) * NEIGHBORS, -1).astype(np.float32)

    oracle_d = oracle_s = oracle_h = None
    if query_truth_sh is not None and query_truth_d is not None:
        oracle_d = v255.best_1d_shifts(query_truth_d, neighbour_d, D_BOUND)
        oracle_s, oracle_h = v255.best_2d_shifts(
            query_truth_sh, neighbour_sh, S_BOUND, H_BOUND
        )
    return PairData(
        features=features,
        neighbour_ids=absolute,
        weights=weight.astype(np.float32),
        raw_sh=raw_sh,
        raw_d=raw_d,
        self_d=self_d,
        self_s=self_s,
        self_h=self_h,
        oracle_d=oracle_d,
        oracle_s=oracle_s,
        oracle_h=oracle_h,
    )


def class_weights(labels: np.ndarray, classes: int) -> torch.Tensor:
    count = np.bincount(labels, minlength=classes).astype(np.float64)
    weight = 1.0 / np.sqrt(np.maximum(count, 1.0))
    weight /= np.mean(weight)
    return torch.tensor(weight, dtype=torch.float32)


def train_model(
    pair: PairData, seed: int, device: torch.device
) -> tuple[ShiftNet, np.ndarray, np.ndarray, dict[str, object]]:
    if pair.oracle_d is None or pair.oracle_s is None or pair.oracle_h is None:
        raise ValueError("training pair labels are required")
    x = pair.features
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    x = ((x - mean) / std).astype(np.float32)
    yd = (pair.oracle_d.reshape(-1) + D_BOUND).astype(np.int64)
    ys = (pair.oracle_s.reshape(-1) + S_BOUND).astype(np.int64)
    yh = (pair.oracle_h.reshape(-1) + H_BOUND).astype(np.int64)
    torch.manual_seed(seed)
    model = ShiftNet(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.4e-3, weight_decay=2e-4)
    loss_d = nn.CrossEntropyLoss(weight=class_weights(yd, 2 * D_BOUND + 1).to(device))
    loss_s = nn.CrossEntropyLoss(weight=class_weights(ys, 2 * S_BOUND + 1).to(device))
    loss_h = nn.CrossEntropyLoss(weight=class_weights(yh, 2 * H_BOUND + 1).to(device))
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(x))
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            xb = torch.from_numpy(x[ids]).to(device)
            d_logit, s_logit, h_logit = model(xb)
            loss = (
                loss_d(d_logit, torch.from_numpy(yd[ids]).to(device))
                + loss_s(s_logit, torch.from_numpy(ys[ids]).to(device))
                + loss_h(h_logit, torch.from_numpy(yh[ids]).to(device))
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(ids)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            history.append({"epoch": epoch, "loss": total_loss / len(x)})
    return model, mean, std, {"history": history, "feature_dim": int(x.shape[1])}


@torch.no_grad()
def predict_probabilities(
    model: ShiftNet,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output: list[list[np.ndarray]] = [[], [], []]
    normalized = ((features - mean) / std).astype(np.float32)
    for start in range(0, len(normalized), BATCH_SIZE):
        x = torch.from_numpy(normalized[start : start + BATCH_SIZE]).to(device)
        logits = model(x)
        for head, value in enumerate(logits):
            output[head].append(torch.softmax(value, dim=-1).cpu().numpy())
    return tuple(np.concatenate(part, axis=0) for part in output)  # type: ignore[return-value]


def choose_shift(
    probability: np.ndarray,
    self_shift: np.ndarray,
    bound: int,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    shape = self_shift.shape
    prob = probability.reshape(*shape, probability.shape[-1])
    if source == "direct":
        class_id = np.argmax(prob, axis=-1)
        return (class_id - bound).astype(np.int16), np.max(prob, axis=-1)
    if source == "self_gate":
        class_id = np.clip(self_shift + bound, 0, 2 * bound).astype(np.int64)
        confidence = np.take_along_axis(prob, class_id[..., None], axis=-1)[..., 0]
        return self_shift.astype(np.int16), confidence
    raise ValueError(source)


def transported_target(
    pair: PairData,
    sh: np.ndarray,
    d: np.ndarray,
    probabilities: tuple[np.ndarray, np.ndarray, np.ndarray],
    source: str,
    mode: str,
    threshold: float,
    blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    d_shift, d_conf = choose_shift(
        probabilities[0], pair.self_d, D_BOUND, source
    )
    s_shift, s_conf = choose_shift(
        probabilities[1], pair.self_s, S_BOUND, source
    )
    h_shift, h_conf = choose_shift(
        probabilities[2], pair.self_h, H_BOUND, source
    )
    d_shift[d_conf < threshold] = 0
    sh_conf = np.minimum(s_conf, h_conf)
    s_shift[sh_conf < threshold] = 0
    h_shift[sh_conf < threshold] = 0
    neighbour_sh = sh[pair.neighbour_ids]
    neighbour_d = d[pair.neighbour_ids]
    aligned_sh = weighted_average(
        v255.apply_2d_shifts(neighbour_sh, s_shift, h_shift), pair.weights
    )
    aligned_d = weighted_average(
        v255.apply_1d_shifts(neighbour_d, d_shift), pair.weights
    )
    pred_sh = pair.raw_sh
    pred_d = pair.raw_d
    if mode in {"sh", "joint"}:
        pred_sh = normalize((1.0 - blend) * pair.raw_sh + blend * aligned_sh)
    if mode in {"d", "joint"}:
        pred_d = normalize((1.0 - blend) * pair.raw_d + blend * aligned_d)
    return pred_sh, pred_d


def label_diagnostics(
    pair: PairData,
    probabilities: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    if pair.oracle_d is None or pair.oracle_s is None or pair.oracle_h is None:
        return {}
    result: dict[str, object] = {}
    for name, probability, truth, self_shift, bound in (
        ("d", probabilities[0], pair.oracle_d, pair.self_d, D_BOUND),
        ("s", probabilities[1], pair.oracle_s, pair.self_s, S_BOUND),
        ("h", probabilities[2], pair.oracle_h, pair.self_h, H_BOUND),
    ):
        pred = np.argmax(probability, axis=-1).reshape(truth.shape) - bound
        result[name] = {
            "direct_accuracy": float(np.mean(pred == truth)),
            "zero_accuracy": float(np.mean(truth == 0)),
            "self_accuracy": float(np.mean(self_shift == truth)),
            "mean_confidence": float(np.mean(np.max(probability, axis=-1))),
        }
    return result


def main() -> None:
    for path in (TARGET_CACHE, SCENE_CACHE, GEODESIC_CACHE, HEIGHT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    data = np.load(TARGET_CACHE)
    sh = data["sh"].astype(np.float32)
    d = data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(data["valid"].astype(bool)).astype(np.int64)
    train_node, test_node = build_node_features(train_pos, test_pos)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_names = [
        f"{source}_{mode}_t{threshold:g}_b{blend:g}"
        for source in SOURCES
        for mode in MODES
        for threshold in THRESHOLDS
        for blend in BLENDS
    ]
    config_results: dict[str, list[dict[str, float]]] = {
        name: [] for name in config_names
    }
    raw_results: list[dict[str, float]] = []
    fold_reports: list[dict[str, object]] = []
    for fold_index, seed in enumerate(FOLD_SEEDS):
        matched = v254.choose_seeded_matched_indices(train_pos, test_pos, VAL_COUNT, seed)
        val_ids = np.intersect1d(matched, valid_ids)
        fit_ids = np.setdiff1d(valid_ids, val_ids)
        train_pair = build_pair_data(
            train_pos[fit_ids],
            train_node[fit_ids],
            fit_ids,
            train_pos,
            train_node,
            sh,
            d,
            sh[fit_ids],
            d[fit_ids],
            fit_ids,
        )
        model, mean, std, training = train_model(
            train_pair, VERSION_NUMBER * 100 + fold_index, device
        )
        val_pair = build_pair_data(
            train_pos[val_ids],
            train_node[val_ids],
            fit_ids,
            train_pos,
            train_node,
            sh,
            d,
            sh[val_ids],
            d[val_ids],
            val_ids,
        )
        probability = predict_probabilities(model, val_pair.features, mean, std, device)
        raw_results.append(metrics(val_pair.raw_sh, val_pair.raw_d, sh[val_ids], d[val_ids]))
        for source in SOURCES:
            for mode in MODES:
                for threshold in THRESHOLDS:
                    for blend in BLENDS:
                        name = f"{source}_{mode}_t{threshold:g}_b{blend:g}"
                        pred_sh, pred_d = transported_target(
                            val_pair,
                            sh,
                            d,
                            probability,
                            source,
                            mode,
                            threshold,
                            blend,
                        )
                        config_results[name].append(
                            metrics(pred_sh, pred_d, sh[val_ids], d[val_ids])
                        )
        fold_reports.append(
            {
                "fold": fold_index,
                "seed": seed,
                "fit_count": int(len(fit_ids)),
                "validation_count": int(len(val_ids)),
                "train_pairs": int(len(train_pair.features)),
                "validation_pairs": int(len(val_pair.features)),
                "training": training,
                "label_diagnostics": label_diagnostics(val_pair, probability),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"v257 official fold {fold_index + 1}/{len(FOLD_SEEDS)}", flush=True)

    raw_mean = {
        key: float(np.mean([row[key] for row in raw_results]))
        for key in ("sh", "d", "weighted")
    }
    summary: dict[str, dict[str, object]] = {}
    for name, rows in config_results.items():
        weighted = np.asarray([row["weighted"] for row in rows])
        summary[name] = {
            "mean": {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("sh", "d", "weighted")
            },
            "std_weighted": float(np.std(weighted)),
            "folds": rows,
        }
    best_name = max(summary, key=lambda name: float(summary[name]["mean"]["weighted"]))
    raw_fold = np.asarray([row["weighted"] for row in raw_results])
    best_fold = np.asarray([row["weighted"] for row in config_results[best_name]])
    official_delta = float(best_fold.mean() - raw_fold.mean())
    official_wins = int(np.sum(best_fold > raw_fold))
    official_selected = bool(official_delta >= MIN_OFFICIAL_DELTA and official_wins >= 3)

    local_score = None
    actions = None
    final_training = None
    test_shift_diagnostics = None
    checkpoint = None
    if official_selected:
        train_pair = build_pair_data(
            train_pos[valid_ids],
            train_node[valid_ids],
            valid_ids,
            train_pos,
            train_node,
            sh,
            d,
            sh[valid_ids],
            d[valid_ids],
            valid_ids,
        )
        model, mean, std, final_training = train_model(
            train_pair, VERSION_NUMBER * 100 + 91, device
        )
        checkpoint = CHECKPOINT_DIR / "seed_25791.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_mean": mean,
                "feature_std": std,
                "feature_dim": int(train_pair.features.shape[1]),
                "config": best_name,
            },
            checkpoint,
        )
        test_pair = build_pair_data(
            test_pos,
            test_node,
            valid_ids,
            train_pos,
            train_node,
            sh,
            d,
            None,
            None,
            None,
        )
        probability = predict_probabilities(model, test_pair.features, mean, std, device)
        if best_name.startswith("self_gate_"):
            source = "self_gate"
            remainder = best_name[len("self_gate_") :]
        elif best_name.startswith("direct_"):
            source = "direct"
            remainder = best_name[len("direct_") :]
        else:
            raise ValueError(best_name)
        mode, threshold_text, blend_text = remainder.split("_")
        threshold = float(threshold_text[1:])
        blend = float(blend_text[1:])
        test_sh, test_d = transported_target(
            test_pair,
            sh,
            d,
            probability,
            source,
            mode,
            threshold,
            blend,
        )
        np.savez_compressed(
            TARGETS,
            sh=test_sh,
            d=test_d,
            raw_sh=test_pair.raw_sh,
            raw_d=test_pair.raw_d,
            method=np.asarray(best_name),
        )
        actions = v247.project(test_sh, test_d, OUTPUT)
        np.savez_compressed(ACTIONS, **actions)
        local_score = v208.score_prediction(OUTPUT)
        test_shift_diagnostics = {
            "d_confidence": summarize(np.max(probability[0], axis=-1)),
            "s_confidence": summarize(np.max(probability[1], axis=-1)),
            "h_confidence": summarize(np.max(probability[2], axis=-1)),
        }

    ranking = sorted(
        summary,
        key=lambda name: float(summary[name]["mean"]["weighted"]),
        reverse=True,
    )
    report = {
        "method": VERSION,
        "status": (
            "gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) > LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "A Physical-AI pair network can predict the small D/S/H transport that the v255 "
            "oracle identified, using only official geometry and observable neighbour evidence."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H pairwise shift labels",
                "official train/test coordinates",
                "official point-cloud and geodesic features",
                "PyTorch three-head shift network",
                "strict v239 official-data carrier after official selection",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher output",
                "local replay truth for training or method selection",
                "hidden labels",
                "raw RT H",
            ],
            "local_truth_usage": (
                "one final score after official five-fold selection"
                if official_selected
                else "not read because official selection failed"
            ),
        },
        "parameters": {
            "fold_seeds": list(FOLD_SEEDS),
            "neighbours": NEIGHBORS,
            "power": POWER,
            "bounds": {"d": D_BOUND, "s": S_BOUND, "h": H_BOUND},
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "thresholds": list(THRESHOLDS),
            "blends": list(BLENDS),
            "modes": list(MODES),
            "sources": list(SOURCES),
            "device": str(device),
        },
        "folds": fold_reports,
        "selection": {
            "raw_mean": raw_mean,
            "best_name": best_name,
            "best": summary[best_name],
            "official_delta": official_delta,
            "official_wins": official_wins,
            "official_selected": official_selected,
            "top_20": [{"name": name, **summary[name]} for name in ranking[:20]],
        },
        "final_training": final_training,
        "test_shift_diagnostics": test_shift_diagnostics,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in actions.items()}
            if actions is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(
                    float(local_score["score_traceback_leaked"]) > LOCAL_GATE
                ),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "targets": PATHS.relative(TARGETS) if TARGETS.is_file() else None,
            "actions": PATHS.relative(ACTIONS) if ACTIONS.is_file() else None,
            "checkpoint": PATHS.relative(checkpoint) if checkpoint is not None else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
