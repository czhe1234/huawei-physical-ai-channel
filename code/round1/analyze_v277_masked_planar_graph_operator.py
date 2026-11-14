#!/usr/bin/env python3
"""Masked transductive graph neural operator for planar spectrum components."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v273_planar_barrier_kernel as v273
import analyze_v274_planar_barrier_attention as v274
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 277
VERSION = "v277_masked_planar_graph_operator"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v277_graph_operator_targets.npz"
OUTPUT = OUT_DIR / "v277_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

LOCAL_GATE = 0.645604
OUTER_SEEDS = (277, 1277, 2277, 3277, 4277)
OUTER_COUNT = 320
INNER_COUNT = 240
SH_COMPONENTS = 56
D_COMPONENTS = 28
HIDDEN = 144
LAYERS = 5
EPOCHS = 80
EVAL_EVERY = 5
MASK_FRACTION = 0.20
MIN_OFFICIAL_DELTA = 5e-4
FINAL_SEEDS = (27701, 27731, 27761)
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


@dataclass
class SpectrumBasis:
    sh_mean: np.ndarray
    sh_components: np.ndarray
    sh_coeff_mean: np.ndarray
    sh_coeff_std: np.ndarray
    d_mean: np.ndarray
    d_components: np.ndarray
    d_coeff_mean: np.ndarray
    d_coeff_std: np.ndarray
    sh_shape: tuple[int, ...]
    d_shape: tuple[int, ...]

    @property
    def dim(self) -> int:
        return self.sh_components.shape[0] + self.d_components.shape[0]


@dataclass
class GraphData:
    node: np.ndarray
    edge: np.ndarray
    neighbors: np.ndarray


def fit_basis(
    sh: np.ndarray,
    d: np.ndarray,
    fit_ids: np.ndarray,
) -> tuple[SpectrumBasis, np.ndarray, dict[str, object]]:
    sqrt_sh = np.sqrt(np.maximum(sh.reshape(len(sh), -1), 0.0)).astype(np.float32)
    sqrt_d = np.sqrt(np.maximum(d.reshape(len(d), -1), 0.0)).astype(np.float32)
    pca_sh = PCA(
        n_components=min(SH_COMPONENTS, len(fit_ids) - 1),
        svd_solver="randomized",
        random_state=277,
    ).fit(sqrt_sh[fit_ids])
    pca_d = PCA(
        n_components=min(D_COMPONENTS, len(fit_ids) - 1),
        svd_solver="randomized",
        random_state=1277,
    ).fit(sqrt_d[fit_ids])
    coeff_sh = pca_sh.transform(sqrt_sh).astype(np.float32)
    coeff_d = pca_d.transform(sqrt_d).astype(np.float32)
    sh_coeff_mean = coeff_sh[fit_ids].mean(axis=0).astype(np.float32)
    sh_coeff_std = np.maximum(coeff_sh[fit_ids].std(axis=0), 1e-4).astype(np.float32)
    d_coeff_mean = coeff_d[fit_ids].mean(axis=0).astype(np.float32)
    d_coeff_std = np.maximum(coeff_d[fit_ids].std(axis=0), 1e-4).astype(np.float32)
    coeff = np.concatenate(
        [
            (coeff_sh - sh_coeff_mean) / sh_coeff_std,
            (coeff_d - d_coeff_mean) / d_coeff_std,
        ],
        axis=1,
    ).astype(np.float32)
    basis = SpectrumBasis(
        sh_mean=pca_sh.mean_.astype(np.float32),
        sh_components=pca_sh.components_.astype(np.float32),
        sh_coeff_mean=sh_coeff_mean,
        sh_coeff_std=sh_coeff_std,
        d_mean=pca_d.mean_.astype(np.float32),
        d_components=pca_d.components_.astype(np.float32),
        d_coeff_mean=d_coeff_mean,
        d_coeff_std=d_coeff_std,
        sh_shape=tuple(sh.shape[1:]),
        d_shape=tuple(d.shape[1:]),
    )
    reconstructed_sh, reconstructed_d = decode_numpy(coeff[fit_ids], basis)
    report = {
        "sh_components": int(len(pca_sh.components_)),
        "d_components": int(len(pca_d.components_)),
        "sh_explained_variance": float(np.sum(pca_sh.explained_variance_ratio_)),
        "d_explained_variance": float(np.sum(pca_d.explained_variance_ratio_)),
        "fit_reconstruction": metrics(
            reconstructed_sh,
            reconstructed_d,
            sh[fit_ids],
            d[fit_ids],
        ),
    }
    return basis, coeff, report


def decode_numpy(
    coeff: np.ndarray,
    basis: SpectrumBasis,
) -> tuple[np.ndarray, np.ndarray]:
    sh_dim = basis.sh_components.shape[0]
    sh_coeff = (
        coeff[:, :sh_dim] * basis.sh_coeff_std + basis.sh_coeff_mean
    )
    d_coeff = (
        coeff[:, sh_dim:] * basis.d_coeff_std + basis.d_coeff_mean
    )
    sqrt_sh = np.maximum(
        basis.sh_mean + sh_coeff @ basis.sh_components, 0.0
    )
    sqrt_d = np.maximum(
        basis.d_mean + d_coeff @ basis.d_components, 0.0
    )
    return (
        normalize((sqrt_sh**2).reshape((len(coeff),) + basis.sh_shape)),
        normalize((sqrt_d**2).reshape((len(coeff),) + basis.d_shape)),
    )


def graph_data(
    node_raw: np.ndarray,
    raw_graph: v274.RawSet,
    fit_ids: np.ndarray,
) -> GraphData:
    node_mean = node_raw[fit_ids].mean(axis=0).astype(np.float32)
    node_std = np.maximum(node_raw[fit_ids].std(axis=0), 1e-5).astype(np.float32)
    edge_values = raw_graph.edge[fit_ids].reshape(-1, raw_graph.edge.shape[-1])
    edge_mean = edge_values.mean(axis=0).astype(np.float32)
    edge_std = np.maximum(edge_values.std(axis=0), 1e-5).astype(np.float32)
    return GraphData(
        node=np.clip((node_raw - node_mean) / node_std, -8.0, 8.0).astype(np.float32),
        edge=np.clip((raw_graph.edge - edge_mean) / edge_std, -8.0, 8.0).astype(np.float32),
        neighbors=raw_graph.neighbor_ids.astype(np.int64),
    )


class MessageLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.edge = nn.Sequential(
            nn.Linear(edge_dim, 96), nn.SiLU(), nn.Linear(96, hidden)
        )
        self.edge_score = nn.Linear(hidden, 1)
        self.update = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        observed: torch.Tensor,
        neighbors: torch.Tensor,
        edge: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_hidden = hidden[neighbors]
        query = self.query(hidden)[:, None, :]
        key = self.key(neighbor_hidden)
        edge_hidden = self.edge(edge)
        score = torch.sum(query * key, dim=2) / math.sqrt(hidden.shape[1])
        score = score + self.edge_score(edge_hidden).squeeze(2)
        attention = torch.softmax(score, dim=1)
        value = self.value(neighbor_hidden) + edge_hidden
        aggregate = torch.sum(attention[..., None] * value, dim=1)
        delta = self.update(torch.cat([hidden, aggregate, observed], dim=1))
        return self.norm(hidden + delta)


class MaskedGraphOperator(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, coeff_dim: int):
        super().__init__()
        self.geometry = nn.Sequential(
            nn.Linear(node_dim, 320),
            nn.SiLU(),
            nn.LayerNorm(320),
            nn.Linear(320, HIDDEN),
        )
        self.observation = nn.Sequential(
            nn.Linear(coeff_dim + 1, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, HIDDEN),
        )
        self.layers = nn.ModuleList(
            [MessageLayer(HIDDEN, edge_dim) for _ in range(LAYERS)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, 256), nn.SiLU(), nn.Linear(256, coeff_dim)
        )

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        neighbors: torch.Tensor,
        observed_coeff: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        observed = self.observation(
            torch.cat([observed_coeff, observed_mask[:, None]], dim=1)
        )
        hidden = self.geometry(node) + observed
        for layer in self.layers:
            hidden = layer(hidden, observed, neighbors, edge)
        return self.output(hidden)


class TorchDecoder(nn.Module):
    def __init__(self, basis: SpectrumBasis):
        super().__init__()
        self.sh_dim = basis.sh_components.shape[0]
        self.register_buffer("sh_mean", torch.from_numpy(basis.sh_mean))
        self.register_buffer("sh_components", torch.from_numpy(basis.sh_components))
        self.register_buffer("sh_coeff_mean", torch.from_numpy(basis.sh_coeff_mean))
        self.register_buffer("sh_coeff_std", torch.from_numpy(basis.sh_coeff_std))
        self.register_buffer("d_mean", torch.from_numpy(basis.d_mean))
        self.register_buffer("d_components", torch.from_numpy(basis.d_components))
        self.register_buffer("d_coeff_mean", torch.from_numpy(basis.d_coeff_mean))
        self.register_buffer("d_coeff_std", torch.from_numpy(basis.d_coeff_std))

    def forward(self, coeff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sh_coeff = coeff[:, : self.sh_dim] * self.sh_coeff_std + self.sh_coeff_mean
        d_coeff = coeff[:, self.sh_dim :] * self.d_coeff_std + self.d_coeff_mean
        sqrt_sh = F.relu(self.sh_mean + sh_coeff @ self.sh_components)
        sqrt_d = F.relu(self.d_mean + d_coeff @ self.d_components)
        sh = sqrt_sh.square()
        d = sqrt_d.square()
        sh /= sh.sum(dim=1, keepdim=True).clamp_min(1e-30)
        d /= d.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return sh, d


@torch.no_grad()
def predict(
    model: MaskedGraphOperator,
    data: GraphData,
    coeff: np.ndarray,
    observed_ids: np.ndarray,
    basis: SpectrumBasis,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    node = torch.from_numpy(data.node).to(device)
    edge = torch.from_numpy(data.edge).to(device)
    neighbors = torch.from_numpy(data.neighbors).to(device)
    observed_coeff = torch.zeros((len(data.node), basis.dim), dtype=torch.float32, device=device)
    observed_mask = torch.zeros(len(data.node), dtype=torch.float32, device=device)
    observed_coeff[observed_ids] = torch.from_numpy(coeff[observed_ids]).to(device)
    observed_mask[observed_ids] = 1.0
    predicted = model(node, edge, neighbors, observed_coeff, observed_mask)
    predicted_np = predicted.cpu().numpy().astype(np.float32)
    pred_sh, pred_d = decode_numpy(predicted_np, basis)
    return pred_sh, pred_d, predicted_np


def train_model(
    data: GraphData,
    basis: SpectrumBasis,
    coeff: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray | None,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[MaskedGraphOperator, dict[str, object]]:
    set_seed(seed)
    model = MaskedGraphOperator(data.node.shape[1], data.edge.shape[2], basis.dim).to(device)
    decoder = TorchDecoder(basis).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    node = torch.from_numpy(data.node).to(device)
    edge = torch.from_numpy(data.edge).to(device)
    neighbors = torch.from_numpy(data.neighbors).to(device)
    coeff_tensor = torch.from_numpy(coeff).to(device)
    sh_flat = torch.from_numpy(sh.reshape(len(sh), -1)).to(device)
    d_flat = torch.from_numpy(d.reshape(len(d), -1)).to(device)
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        mask_count = max(int(round(MASK_FRACTION * len(fit_ids))), 64)
        masked_ids = rng.choice(fit_ids, size=mask_count, replace=False)
        observed_ids = np.setdiff1d(fit_ids, masked_ids)
        observed_coeff = torch.zeros(
            (len(data.node), basis.dim), dtype=torch.float32, device=device
        )
        observed_mask = torch.zeros(len(data.node), dtype=torch.float32, device=device)
        observed_coeff[observed_ids] = coeff_tensor[observed_ids]
        observed_mask[observed_ids] = 1.0
        predicted = model(node, edge, neighbors, observed_coeff, observed_mask)
        masked = torch.from_numpy(masked_ids).to(device)
        pred_coeff = predicted[masked]
        pred_sh, pred_d = decoder(pred_coeff)
        truth_sh = sh_flat[masked]
        truth_d = d_flat[masked]
        sh_loss = 1.0 - F.cosine_similarity(pred_sh, truth_sh, dim=1).mean()
        d_loss = 1.0 - F.cosine_similarity(pred_d, truth_d, dim=1).mean()
        coeff_loss = F.smooth_l1_loss(pred_coeff, coeff_tensor[masked], beta=0.8)
        loss = 0.50 * sh_loss + 0.46 * d_loss + 0.04 * coeff_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs:
            item: dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
            }
            if validation_ids is not None:
                pred_val_sh, pred_val_d, _ = predict(
                    model, data, coeff, fit_ids, basis, device
                )
                val_metric = metrics(
                    pred_val_sh[validation_ids],
                    pred_val_d[validation_ids],
                    sh[validation_ids],
                    d[validation_ids],
                )
                item["validation"] = val_metric
                if val_metric["weighted"] > best_metric:
                    best_metric = val_metric["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"v277 seed={seed} epoch={epoch}/{epochs} loss={float(loss.detach().cpu()):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": float(best_metric) if np.isfinite(best_metric) else None,
        "history": history,
    }


def choose_residual_alpha(
    base: np.ndarray, model: np.ndarray, truth: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        blended = normalize(base + alpha * (model - base))
        rows.append(
            {
                "alpha": float(alpha),
                "cosine": float(np.mean(row_cos(blended, truth))),
            }
        )
    best = max(rows, key=lambda row: row["cosine"])
    return float(best["alpha"]), rows


def main() -> None:
    for path in (
        BASE,
        TARGET_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = np.load(TARGET_CACHE)
    sh = targets["sh"].astype(np.float32)
    d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    (
        all_pos,
        signatures,
        blocker_count,
        scene_features,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = v274.load_geometry()
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    all_ids = np.arange(len(all_pos), dtype=np.int64)
    raw_graph = v274.build_raw_set(
        all_ids,
        all_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )

    outer_rows: list[dict[str, object]] = []
    outer_base_sh: list[np.ndarray] = []
    outer_base_d: list[np.ndarray] = []
    outer_model_sh: list[np.ndarray] = []
    outer_model_d: list[np.ndarray] = []
    outer_truth_sh: list[np.ndarray] = []
    outer_truth_d: list[np.ndarray] = []
    selected_epochs: list[int] = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = v273.matched_subset(
            outer_fit,
            train_pos[:, :2],
            test_pos[:, :2],
            INNER_COUNT,
            seed + 10_000,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)

        inner_basis, inner_coeff, inner_basis_report = fit_basis(
            sh, d, inner_fit
        )
        inner_graph = graph_data(node_raw, raw_graph, inner_fit)
        _inner_model, inner_training = train_model(
            inner_graph,
            inner_basis,
            inner_coeff,
            sh,
            d,
            inner_fit,
            inner_ids,
            seed,
            EPOCHS,
            device,
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)

        outer_basis, outer_coeff, outer_basis_report = fit_basis(
            sh, d, outer_fit
        )
        outer_graph = graph_data(node_raw, raw_graph, outer_fit)
        outer_model, outer_training = train_model(
            outer_graph,
            outer_basis,
            outer_coeff,
            sh,
            d,
            outer_fit,
            None,
            seed + 50_000,
            selected_epoch,
            device,
        )
        pred_sh, pred_d, _ = predict(
            outer_model,
            outer_graph,
            outer_coeff,
            outer_fit,
            outer_basis,
            device,
        )
        base_sh, base_d = v254.euclidean_predict(
            train_pos[outer_fit, :2],
            train_pos[outer_ids, :2],
            sh[outer_fit],
            d[outer_fit],
            12,
            2.0,
        )
        baseline = metrics(
            base_sh, base_d, sh[outer_ids], d[outer_ids]
        )
        model_metric = metrics(
            pred_sh[outer_ids], pred_d[outer_ids], sh[outer_ids], d[outer_ids]
        )
        delta = {
            key: model_metric[key] - baseline[key]
            for key in ("sh", "d", "weighted")
        }
        outer_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "selected_epoch": selected_epoch,
                "inner_basis": inner_basis_report,
                "inner_training": inner_training,
                "outer_basis": outer_basis_report,
                "outer_training": outer_training,
                "baseline": baseline,
                "model": model_metric,
                "delta": delta,
            }
        )
        outer_base_sh.append(base_sh)
        outer_base_d.append(base_d)
        outer_model_sh.append(pred_sh[outer_ids])
        outer_model_d.append(pred_d[outer_ids])
        outer_truth_sh.append(sh[outer_ids])
        outer_truth_d.append(d[outer_ids])
        print(
            f"v277 outer {fold + 1}/{len(OUTER_SEEDS)} epoch={selected_epoch} "
            f"baseline={baseline['weighted']:.6f} model={model_metric['weighted']:.6f} "
            f"delta={delta['weighted']:+.6f}",
            flush=True,
        )

    fold_delta = np.asarray(
        [row["delta"]["weighted"] for row in outer_rows], dtype=np.float64
    )
    fold_sh_delta = np.asarray(
        [row["delta"]["sh"] for row in outer_rows], dtype=np.float64
    )
    fold_d_delta = np.asarray(
        [row["delta"]["d"] for row in outer_rows], dtype=np.float64
    )
    official_selected = bool(
        np.mean(fold_delta) >= MIN_OFFICIAL_DELTA
        and np.sum(fold_delta > 0.0) >= 3
        and np.mean(fold_sh_delta) > 0.0
        and np.mean(fold_d_delta) > 0.0
    )
    cv_base_sh = np.concatenate(outer_base_sh)
    cv_base_d = np.concatenate(outer_base_d)
    cv_model_sh = np.concatenate(outer_model_sh)
    cv_model_d = np.concatenate(outer_model_d)
    cv_truth_sh = np.concatenate(outer_truth_sh)
    cv_truth_d = np.concatenate(outer_truth_d)
    alpha_sh, alpha_sh_grid = choose_residual_alpha(
        cv_base_sh, cv_model_sh, cv_truth_sh
    )
    alpha_d, alpha_d_grid = choose_residual_alpha(
        cv_base_d, cv_model_d, cv_truth_d
    )

    final_training: list[dict[str, object]] = []
    local_score = None
    projection_diagnostics = None
    if official_selected:
        final_epoch = max(int(np.median(selected_epochs)), 1)
        full_basis, full_coeff, full_basis_report = fit_basis(sh, d, valid_ids)
        full_graph = graph_data(node_raw, raw_graph, valid_ids)
        test_model_sh_parts = []
        test_model_d_parts = []
        for seed in FINAL_SEEDS:
            model, training = train_model(
                full_graph,
                full_basis,
                full_coeff,
                sh,
                d,
                valid_ids,
                None,
                seed,
                final_epoch,
                device,
            )
            pred_sh, pred_d, _ = predict(
                model,
                full_graph,
                full_coeff,
                valid_ids,
                full_basis,
                device,
            )
            test_model_sh_parts.append(pred_sh[len(train_pos) :])
            test_model_d_parts.append(pred_d[len(train_pos) :])
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "model_state": model.state_dict(),
                    "basis": full_basis,
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
        test_model_sh = normalize(np.mean(test_model_sh_parts, axis=0))
        test_model_d = normalize(np.mean(test_model_d_parts, axis=0))
        test_base_sh, test_base_d = v254.euclidean_predict(
            train_pos[valid_ids, :2],
            test_pos[:, :2],
            sh[valid_ids],
            d[valid_ids],
            12,
            2.0,
        )
        current_sh, current_d = v274.extract_current_marginals(BASE)
        target_sh = normalize(
            current_sh + alpha_sh * (test_model_sh - test_base_sh)
        )
        target_d = normalize(
            current_d + alpha_d * (test_model_d - test_base_d)
        )
        np.savez_compressed(
            TARGETS_OUT,
            base_sh=test_base_sh,
            base_d=test_base_d,
            model_sh=test_model_sh,
            model_d=test_model_d,
            target_sh=target_sh,
            target_d=target_d,
            alpha_sh=np.asarray(alpha_sh, dtype=np.float32),
            alpha_d=np.asarray(alpha_d, dtype=np.float32),
            basis_report_json=np.asarray(json.dumps(full_basis_report)),
        )
        projection_diagnostics = v247.project(target_sh, target_d, OUTPUT)
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": (
            "local_gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "A masked nonlinear graph operator over all coplanar UE nodes can infer new "
            "delay-angle component combinations that linear spectrum diffusion cannot create."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP labels",
                "official train/test XY coordinates",
                "official point-cloud wall/component/height features",
                "strict v239 official-only carrier",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher outputs",
                "hidden labels",
                "RT MIMO H or path coefficients",
                "local replay truth for training or selection",
            ],
            "local_truth_usage": (
                "one final gate score after nested official selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "sh_components": SH_COMPONENTS,
            "d_components": D_COMPONENTS,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "epochs": EPOCHS,
            "mask_fraction": MASK_FRACTION,
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
            "final_seeds": list(FINAL_SEEDS),
        },
        "outer_folds": outer_rows,
        "official_selection": {
            "mean_delta": {
                "sh": float(np.mean(fold_sh_delta)),
                "d": float(np.mean(fold_d_delta)),
                "weighted": float(np.mean(fold_delta)),
            },
            "weighted_wins": int(np.sum(fold_delta > 0.0)),
            "selected_epochs": selected_epochs,
            "selected": official_selected,
            "residual_alpha_sh": alpha_sh,
            "residual_alpha_d": alpha_d,
            "alpha_sh_grid": alpha_sh_grid,
            "alpha_d_grid": alpha_d_grid,
        },
        "final_training": final_training,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in projection_diagnostics.items()}
            if projection_diagnostics is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(
                    float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
                ),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
