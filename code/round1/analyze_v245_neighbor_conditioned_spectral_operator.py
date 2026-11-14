#!/usr/bin/env python3
"""v245: strict official-data neighbor-conditioned spectral operator.

The model predicts the evaluator-sensitive leaked S-H spectrum and PDP from a
set of nearby official training channels.  Unlike a fixed KNN average, every
neighbor token carries its observed spectrum, geometry, and relative position;
a Transformer learns how the local spectrum changes between the neighbor and
the query.  Training, epoch selection, and action calibration use only official
training channels and a matched spatial holdout.  The local replay truth is
read once after generation as a diagnostic gate.
"""

from __future__ import annotations

import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import analyze_v208_strict_historical_replay as v208
import analyze_v234_official_neighbor_support_attention as v234
import generate_v187_leaked_pas_lifecycle_projection as leak
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v159_frontier_consensus_support_soup import digest
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 245
VERSION = "v245_neighbor_conditioned_spectral_operator"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v245_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v245_test_spectral_targets.npz"
ACTIONS = OUT_DIR / "v245_spectral_operator_actions.npz"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

LOCAL_GATE = 0.645604
FLOOR = 1e-10
VAL_COUNT = 320
K_NEIGHBORS = 40
SH_COMPONENTS = 128
D_COMPONENTS = 48
TRAIN_EPOCHS = 220
BATCH_SIZE = 56
FINAL_SEEDS = (24501, 24529, 24557)
REGIMES = ("clear", "light", "medium", "heavy")
PROJECT_BATCH = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_distribution(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape((np.sum(valid),) + (1,) * (out.ndim - 1))
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


@dataclass
class LogPCABasis:
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray
    shape: tuple[int, ...]


def centered_log(x: np.ndarray) -> np.ndarray:
    flat = np.log(np.maximum(np.asarray(x, dtype=np.float32).reshape(len(x), -1), FLOOR))
    flat -= flat.mean(axis=1, keepdims=True)
    return flat.astype(np.float32)


def fit_log_pca(x: np.ndarray, components: int, seed: int) -> tuple[LogPCABasis, np.ndarray]:
    logits = centered_log(x)
    n_components = min(components, len(logits) - 1, logits.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    coeff = pca.fit_transform(logits).astype(np.float32)
    scale = np.maximum(coeff.std(axis=0), 1e-4).astype(np.float32)
    basis = LogPCABasis(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        scale=scale,
        shape=tuple(x.shape[1:]),
    )
    return basis, (coeff / scale[None, :]).astype(np.float32)


def transform_log_pca(x: np.ndarray, basis: LogPCABasis) -> np.ndarray:
    logits = centered_log(x)
    coeff = (logits - basis.mean[None, :]) @ basis.components.T
    return (coeff / basis.scale[None, :]).astype(np.float32)


def decode_log_pca(z: np.ndarray, basis: LogPCABasis) -> np.ndarray:
    logits = (np.asarray(z, dtype=np.float32) * basis.scale[None, :]) @ basis.components
    logits += basis.mean[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(np.clip(logits, -35.0, 0.0)).astype(np.float32)
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), TINY)
    return prob.reshape((len(prob),) + basis.shape).astype(np.float32)


def geometry_features(
    prefix: str,
    indices: np.ndarray | None,
    pos: np.ndarray,
    blocked: np.ndarray,
) -> np.ndarray:
    scene = np.load(v215.SCENE_CACHE)[f"{prefix}_features"].astype(np.float32)
    geodesic = np.load(v215.GEODESIC_CACHE)[f"{prefix}_features"].astype(np.float32)
    if indices is not None:
        scene = scene[indices]
        geodesic = geodesic[indices]
    path = v234.compact_path_features(prefix, pos, indices)
    regimes = regime_of(blocked).astype(str)
    onehot = np.stack([(regimes == name).astype(np.float32) for name in REGIMES], axis=1)
    return np.nan_to_num(
        np.concatenate(
            [
                v215.position_features(pos),
                blocked[:, None].astype(np.float32) / 160.0,
                onehot,
                scene,
                geodesic,
                path,
            ],
            axis=1,
        ),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    ).astype(np.float32)


@dataclass
class QuerySet:
    query_features: np.ndarray
    neighbor_idx: np.ndarray
    edge_features: np.ndarray
    base_weights: np.ndarray
    anchor_latent: np.ndarray


def make_neighbors(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
    query_ids: np.ndarray | None,
    bank_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = v218.effective_distance(query_pos, query_blocked, bank_pos, bank_blocked)
    if query_ids is not None and bank_ids is not None:
        dist[query_ids[:, None] == bank_ids[None, :]] = np.inf
    k = min(K_NEIGHBORS, len(bank_pos) - (1 if query_ids is not None else 0))
    idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
    dsel = np.take_along_axis(dist, idx, axis=1)
    order = np.argsort(dsel, axis=1)
    idx = np.take_along_axis(idx, order, axis=1).astype(np.int32)
    dsel = np.take_along_axis(dsel, order, axis=1).astype(np.float32)
    weights = 1.0 / np.maximum(dsel, 0.35) ** 2
    same = regime_of(query_blocked)[:, None] == regime_of(bank_blocked)[idx]
    weights *= np.where(same, 1.22, 0.70).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    return idx, dsel, weights.astype(np.float32)


def build_query_set(
    query_features: np.ndarray,
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
    bank_latent: np.ndarray,
    query_ids: np.ndarray | None,
    bank_ids: np.ndarray | None,
) -> QuerySet:
    idx, dsel, weights = make_neighbors(
        query_pos, query_blocked, bank_pos, bank_blocked, query_ids, bank_ids
    )
    edges = v234.edge_features(
        query_pos, query_blocked, bank_pos, bank_blocked, idx, dsel
    )
    edges = np.concatenate(
        [edges, np.log(np.maximum(weights, 1e-12))[..., None]], axis=2
    ).astype(np.float32)
    anchor = np.sum(weights[..., None] * bank_latent[idx], axis=1).astype(np.float32)
    return QuerySet(query_features, idx, edges, weights, anchor)


def standardize(
    bank_features: np.ndarray,
    queries: list[QuerySet],
) -> tuple[np.ndarray, list[QuerySet], dict[str, np.ndarray]]:
    node_mean = bank_features.mean(axis=0).astype(np.float32)
    node_std = np.maximum(bank_features.std(axis=0), 1e-5).astype(np.float32)
    edge_train = queries[0].edge_features.reshape(-1, queries[0].edge_features.shape[2])
    edge_mean = edge_train.mean(axis=0).astype(np.float32)
    edge_std = np.maximum(edge_train.std(axis=0), 1e-5).astype(np.float32)
    bank = np.clip((bank_features - node_mean) / node_std, -8.0, 8.0).astype(np.float32)
    result = []
    for query in queries:
        result.append(
            QuerySet(
                query_features=np.clip(
                    (query.query_features - node_mean) / node_std, -8.0, 8.0
                ).astype(np.float32),
                neighbor_idx=query.neighbor_idx,
                edge_features=np.clip(
                    (query.edge_features - edge_mean) / edge_std, -8.0, 8.0
                ).astype(np.float32),
                base_weights=query.base_weights,
                anchor_latent=query.anchor_latent,
            )
        )
    return bank, result, {
        "node_mean": node_mean,
        "node_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
    }


class SpectralOperator(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, latent_dim: int):
        super().__init__()
        width = 256
        self.query = nn.Sequential(
            nn.Linear(node_dim + latent_dim, 384),
            nn.SiLU(),
            nn.LayerNorm(384),
            nn.Linear(384, width),
        )
        self.neighbor = nn.Sequential(
            nn.Linear(node_dim + edge_dim + latent_dim, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.06),
            nn.Linear(512, width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=8,
            dim_feedforward=768,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 384),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(384, latent_dim),
        )
        self.delta_scale = nn.Parameter(torch.tensor(0.45, dtype=torch.float32))

    def forward(
        self,
        query_features: torch.Tensor,
        bank_features: torch.Tensor,
        bank_latent: torch.Tensor,
        neighbor_idx: torch.Tensor,
        edges: torch.Tensor,
        anchor_latent: torch.Tensor,
        mask_neighbors: bool,
    ) -> torch.Tensor:
        q = self.query(torch.cat([query_features, anchor_latent], dim=1)).unsqueeze(1)
        nfeat = bank_features[neighbor_idx]
        nlatent = bank_latent[neighbor_idx]
        ntoken = self.neighbor(torch.cat([nfeat, edges, nlatent], dim=2))
        tokens = torch.cat([q, ntoken], dim=1)
        key_mask = None
        if mask_neighbors and self.training:
            dropped = torch.rand(neighbor_idx.shape, device=neighbor_idx.device) < 0.16
            dropped[:, 0] = False
            key_mask = torch.cat(
                [
                    torch.zeros((len(dropped), 1), dtype=torch.bool, device=dropped.device),
                    dropped,
                ],
                dim=1,
            )
        encoded = self.encoder(tokens, src_key_padding_mask=key_mask)
        scale = 2.8 * torch.sigmoid(self.delta_scale)
        return anchor_latent + scale * torch.tanh(self.head(encoded[:, 0]))


def torch_basis(
    basis: LogPCABasis, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(basis.scale).to(device),
        torch.from_numpy(basis.components).to(device),
        torch.from_numpy(basis.mean).to(device),
    )


def torch_decode(
    latent: torch.Tensor,
    basis_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    start: int,
    end: int,
) -> torch.Tensor:
    z = latent[:, start:end]
    scale, components, mean = basis_tensors
    logits = (z * scale[None, :]) @ components + mean[None, :]
    return torch.softmax(logits, dim=1)


def spectral_loss(
    prediction: torch.Tensor,
    target_latent: torch.Tensor,
    target_sh: torch.Tensor,
    target_d: torch.Tensor,
    sh_basis_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    d_basis_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    sh_dim: int,
) -> torch.Tensor:
    pred_sh = torch_decode(prediction, sh_basis_tensors, 0, sh_dim)
    pred_d = torch_decode(
        prediction, d_basis_tensors, sh_dim, prediction.shape[1]
    )
    sh_loss = 1.0 - F.cosine_similarity(pred_sh, target_sh, dim=1).mean()
    d_loss = 1.0 - F.cosine_similarity(pred_d, target_d, dim=1).mean()
    latent_loss = F.smooth_l1_loss(prediction, target_latent, beta=0.65)
    return 0.49 * sh_loss + 0.43 * d_loss + 0.08 * latent_loss


@torch.no_grad()
def predict_latent(
    model: SpectralOperator,
    query: QuerySet,
    bank_features: np.ndarray,
    bank_latent: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    bank_f = torch.from_numpy(bank_features).to(device)
    bank_z = torch.from_numpy(bank_latent).to(device)
    result = []
    for start in range(0, len(query.query_features), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(query.query_features))
        result.append(
            model(
                torch.from_numpy(query.query_features[start:end]).to(device),
                bank_f,
                bank_z,
                torch.from_numpy(query.neighbor_idx[start:end]).to(device),
                torch.from_numpy(query.edge_features[start:end]).to(device),
                torch.from_numpy(query.anchor_latent[start:end]).to(device),
                False,
            ).cpu().numpy()
        )
    return np.concatenate(result).astype(np.float32)


def evaluate_spectra(
    latent: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
    sh_basis: LogPCABasis,
    d_basis: LogPCABasis,
) -> dict[str, float]:
    sh_end = len(sh_basis.scale)
    pred_sh = decode_log_pca(latent[:, :sh_end], sh_basis)
    pred_d = decode_log_pca(latent[:, sh_end:], d_basis)
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def train_model(
    query: QuerySet,
    target_latent: np.ndarray,
    target_sh: np.ndarray,
    target_d: np.ndarray,
    bank_features: np.ndarray,
    bank_latent: np.ndarray,
    sh_basis: LogPCABasis,
    d_basis: LogPCABasis,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[QuerySet, np.ndarray, np.ndarray] | None,
) -> tuple[SpectralOperator, dict[str, object]]:
    set_seed(seed)
    model = SpectralOperator(
        bank_features.shape[1], query.edge_features.shape[2], target_latent.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.5e-4, weight_decay=3.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    bank_f = torch.from_numpy(bank_features).to(device)
    bank_z = torch.from_numpy(bank_latent).to(device)
    sh_basis_tensors = torch_basis(sh_basis, device)
    d_basis_tensors = torch_basis(d_basis, device)
    sh_dim = len(sh_basis.scale)
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(query.query_features))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            prediction = model(
                torch.from_numpy(query.query_features[ids]).to(device),
                bank_f,
                bank_z,
                torch.from_numpy(query.neighbor_idx[ids]).to(device),
                torch.from_numpy(query.edge_features[ids]).to(device),
                torch.from_numpy(query.anchor_latent[ids]).to(device),
                True,
            )
            truth_z = torch.from_numpy(target_latent[ids]).to(device)
            truth_sh = torch.from_numpy(target_sh[ids].reshape(len(ids), -1)).to(device)
            truth_d = torch.from_numpy(target_d[ids].reshape(len(ids), -1)).to(device)
            loss = spectral_loss(
                prediction,
                truth_z,
                truth_sh,
                truth_d,
                sh_basis_tensors,
                d_basis_tensors,
                sh_dim,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item: dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
            }
            if validation is not None:
                val_query, val_sh, val_d = validation
                val_z = predict_latent(model, val_query, bank_features, bank_latent, device)
                metrics = evaluate_spectra(val_z, val_sh, val_d, sh_basis, d_basis)
                anchor_metrics = evaluate_spectra(
                    val_query.anchor_latent, val_sh, val_d, sh_basis, d_basis
                )
                item["val_model"] = metrics
                item["val_anchor"] = anchor_metrics
                if metrics["weighted"] > best_metric:
                    best_metric = metrics["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
            print(f"v245 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": None if not np.isfinite(best_metric) else float(best_metric),
        "history": history,
    }


def calibrate_alpha(
    anchor_z: np.ndarray,
    model_z: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
    regimes: np.ndarray,
    sh_basis: LogPCABasis,
    d_basis: LogPCABasis,
) -> tuple[dict[str, float], dict[str, object]]:
    sh_end = len(sh_basis.scale)
    anchor_sh = decode_log_pca(anchor_z[:, :sh_end], sh_basis)
    anchor_d = decode_log_pca(anchor_z[:, sh_end:], d_basis)
    model_sh = decode_log_pca(model_z[:, :sh_end], sh_basis)
    model_d = decode_log_pca(model_z[:, sh_end:], d_basis)
    alpha_grid = np.linspace(0.0, 1.15, 24, dtype=np.float32)
    selected: dict[str, float] = {}
    report: dict[str, object] = {}
    for regime in REGIMES:
        mask = regimes.astype(str) == regime
        if np.sum(mask) < 8:
            selected[regime] = 0.0
            report[regime] = {"count": int(np.sum(mask)), "alpha": 0.0}
            continue
        scores = []
        for alpha in alpha_grid:
            sh = normalize_distribution(
                (1.0 - alpha) * anchor_sh[mask] + alpha * model_sh[mask]
            )
            d = normalize_distribution(
                (1.0 - alpha) * anchor_d[mask] + alpha * model_d[mask]
            )
            scores.append(
                0.52 * float(np.mean(row_cos(sh, truth_sh[mask])))
                + 0.48 * float(np.mean(row_cos(d, truth_d[mask])))
            )
        best = int(np.argmax(scores))
        selected[regime] = float(alpha_grid[best])
        report[regime] = {
            "count": int(np.sum(mask)),
            "alpha": float(alpha_grid[best]),
            "anchor_score": float(scores[0]),
            "best_score": float(scores[best]),
        }
    return selected, report


def extract_sh_d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    source = np.load(path, mmap_mode="r")
    sh_parts = []
    d_parts = []
    for start in range(0, len(source), PROJECT_BATCH):
        end = min(start + PROJECT_BATCH, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        leaked_power = leak.leaked_cell_power(leak.leaked_forward(h))
        sh = leaked_power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
        cir = np.fft.ifft(h, axis=-1)
        arr = cir.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
        d = (np.abs(arr).astype(np.float32) ** 2).sum(
            axis=(1, 2, 3, 4), dtype=np.float64
        ).astype(np.float32)
        sh_parts.append(normalize_distribution(sh))
        d_parts.append(normalize_distribution(d))
    return np.concatenate(sh_parts), np.concatenate(d_parts)


def apply_targets(
    target_sh: np.ndarray,
    target_d: np.ndarray,
    alpha_by_regime: dict[str, float],
    regimes: np.ndarray,
) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    alpha = np.asarray([alpha_by_regime[str(r)] for r in regimes], dtype=np.float32)
    diagnostics: dict[str, list[np.ndarray]] = {
        "sh_l1": [],
        "d_l1": [],
        "power_factor": [],
        "sh_gain_min": [],
        "sh_gain_max": [],
        "d_gain_min": [],
        "d_gain_max": [],
    }
    for start in range(0, len(source), PROJECT_BATCH):
        end = min(start + PROJECT_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        power0 = np.maximum(
            np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
        ).astype(np.float32)
        local_alpha = alpha[start:end, None, None]

        spec = leak.leaked_forward(h0)
        spec_power = leak.leaked_cell_power(spec)
        current_sh = normalize_distribution(
            spec_power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
        )
        desired_sh = normalize_distribution(
            (1.0 - local_alpha) * current_sh
            + local_alpha * np.asarray(target_sh[start:end], dtype=np.float32)
        )
        sh_gain = np.sqrt(
            np.maximum(desired_sh, FLOOR) / np.maximum(current_sh, FLOOR)
        ).astype(np.float32)
        sh_gain = np.clip(sh_gain, 0.42, 2.55)
        spec *= sh_gain[:, None, :, :, None, None]
        h1 = leak.leaked_inverse(spec)
        p1 = np.maximum(
            np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
        ).astype(np.float32)
        h1 *= np.sqrt(power0 / p1)[:, None, None, None].astype(np.float32)

        cir = np.fft.ifft(h1, axis=-1)
        arr = cir.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
        arr_power = np.abs(arr).astype(np.float32) ** 2
        current_d = normalize_distribution(
            arr_power.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
        )
        d_alpha = alpha[start:end, None]
        desired_d = normalize_distribution(
            (1.0 - d_alpha) * current_d
            + d_alpha * np.asarray(target_d[start:end], dtype=np.float32)
        )
        d_gain = np.sqrt(
            np.maximum(desired_d, FLOOR) / np.maximum(current_d, FLOOR)
        ).astype(np.float32)
        d_gain = np.clip(d_gain, 0.46, 2.35)
        arr *= d_gain[:, None, None, None, None, :]
        h2 = np.fft.fft(
            arr.reshape(end - start, MP * MH * MV, N_ANT, S_SC), axis=-1
        ).astype(np.complex64)
        p2 = np.maximum(
            np.sum(np.abs(h2) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
        ).astype(np.float32)
        h2 *= np.sqrt(power0 / p2)[:, None, None, None].astype(np.float32)
        if not np.isfinite(h2).all():
            raise FloatingPointError(f"non-finite v245 block {start}:{end}")
        output[start:end] = h2
        output.flush()

        diagnostics["sh_l1"].append(
            np.sum(np.abs(desired_sh - current_sh), axis=(1, 2)).astype(np.float32)
        )
        diagnostics["d_l1"].append(
            np.sum(np.abs(desired_d - current_d), axis=1).astype(np.float32)
        )
        diagnostics["power_factor"].append((p2 / power0).astype(np.float32))
        diagnostics["sh_gain_min"].append(sh_gain.reshape(end - start, -1).min(axis=1))
        diagnostics["sh_gain_max"].append(sh_gain.reshape(end - start, -1).max(axis=1))
        diagnostics["d_gain_min"].append(d_gain.min(axis=1))
        diagnostics["d_gain_max"].append(d_gain.max(axis=1))
        print(f"v245 projection {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diagnostics.items()}


def finite_check(path: Path) -> dict[str, object]:
    arr = np.load(path, mmap_mode="r")
    finite = True
    for start in range(0, len(arr), 12):
        block = np.asarray(arr[start : start + 12], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block).all())
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "finite": finite}


def package(report: dict[str, object]) -> None:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(Path(__file__), Path(__file__).name)
        for checkpoint in sorted(CHECKPOINT_DIR.glob("*.pt")):
            archive.write(checkpoint, checkpoint.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    report["status"] = "candidate_pending"
    report["integrity"] = {
        **finite_check(OUTPUT),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    for path in [BASE, TARGET_CACHE, GEOM_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_data = np.load(TARGET_CACHE)
    sh_all = target_data["sh"].astype(np.float32)
    d_all = target_data["d"].astype(np.float32)
    valid = target_data["valid"].astype(bool)
    valid_ids = np.flatnonzero(valid).astype(np.int64)

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    matched = choose_val_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, "matched-test", VERSION_NUMBER
    )
    val_ids = np.intersect1d(matched, valid_ids)
    train_ids = np.setdiff1d(valid_ids, val_ids)

    sh_basis, train_sh_z = fit_log_pca(sh_all[train_ids], SH_COMPONENTS, 24501)
    d_basis, train_d_z = fit_log_pca(d_all[train_ids], D_COMPONENTS, 24503)
    bank_latent = np.concatenate([train_sh_z, train_d_z], axis=1).astype(np.float32)
    target_latent = bank_latent.copy()
    val_latent = np.concatenate(
        [
            transform_log_pca(sh_all[val_ids], sh_basis),
            transform_log_pca(d_all[val_ids], d_basis),
        ],
        axis=1,
    ).astype(np.float32)

    bank_raw = geometry_features(
        "train", train_ids, train_pos[train_ids], train_blocked[train_ids]
    )
    train_raw = build_query_set(
        bank_raw,
        train_pos[train_ids],
        train_blocked[train_ids],
        train_pos[train_ids],
        train_blocked[train_ids],
        bank_latent,
        train_ids,
        train_ids,
    )
    val_raw = build_query_set(
        geometry_features("train", val_ids, train_pos[val_ids], train_blocked[val_ids]),
        train_pos[val_ids],
        train_blocked[val_ids],
        train_pos[train_ids],
        train_blocked[train_ids],
        bank_latent,
        None,
        None,
    )
    bank_features, [train_query, val_query], split_stats = standardize(
        bank_raw, [train_raw, val_raw]
    )
    validation_model, validation_training = train_model(
        train_query,
        target_latent,
        sh_all[train_ids],
        d_all[train_ids],
        bank_features,
        bank_latent,
        sh_basis,
        d_basis,
        FINAL_SEEDS[0],
        TRAIN_EPOCHS,
        device,
        (val_query, sh_all[val_ids], d_all[val_ids]),
    )
    val_prediction = predict_latent(
        validation_model, val_query, bank_features, bank_latent, device
    )
    validation_metrics = {
        "anchor": evaluate_spectra(
            val_query.anchor_latent, sh_all[val_ids], d_all[val_ids], sh_basis, d_basis
        ),
        "model": evaluate_spectra(
            val_prediction, sh_all[val_ids], d_all[val_ids], sh_basis, d_basis
        ),
    }
    alpha_by_regime, calibration = calibrate_alpha(
        val_query.anchor_latent,
        val_prediction,
        sh_all[val_ids],
        d_all[val_ids],
        regime_of(train_blocked[val_ids]),
        sh_basis,
        d_basis,
    )
    selected_epochs = max(20, int(validation_training["best_epoch"]))
    del validation_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_sh_basis, final_sh_z = fit_log_pca(
        sh_all[valid_ids], SH_COMPONENTS, 24511
    )
    final_d_basis, final_d_z = fit_log_pca(d_all[valid_ids], D_COMPONENTS, 24513)
    final_latent = np.concatenate([final_sh_z, final_d_z], axis=1).astype(np.float32)
    final_bank_raw = geometry_features(
        "train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids]
    )
    final_raw = build_query_set(
        final_bank_raw,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        train_pos[valid_ids],
        train_blocked[valid_ids],
        final_latent,
        valid_ids,
        valid_ids,
    )
    test_raw = build_query_set(
        geometry_features("test", None, test_pos, test_blocked),
        test_pos,
        test_blocked,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        final_latent,
        None,
        None,
    )
    final_bank_features, [final_query, test_query], final_stats = standardize(
        final_bank_raw, [final_raw, test_raw]
    )
    seed_sh = []
    seed_d = []
    final_training = []
    for seed in FINAL_SEEDS:
        model, info = train_model(
            final_query,
            final_latent,
            sh_all[valid_ids],
            d_all[valid_ids],
            final_bank_features,
            final_latent,
            final_sh_basis,
            final_d_basis,
            seed,
            selected_epochs,
            device,
            None,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "epochs": selected_epochs,
                "feature_stats": final_stats,
            },
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        prediction = predict_latent(
            model, test_query, final_bank_features, final_latent, device
        )
        sh_end = len(final_sh_basis.scale)
        seed_sh.append(decode_log_pca(prediction[:, :sh_end], final_sh_basis))
        seed_d.append(decode_log_pca(prediction[:, sh_end:], final_d_basis))
        final_training.append(info)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_sh = normalize_distribution(np.mean(np.stack(seed_sh), axis=0))
    test_d = normalize_distribution(np.mean(np.stack(seed_d), axis=0))
    np.savez_compressed(TARGETS, sh=test_sh, d=test_d)

    base_sh, base_d = extract_sh_d(BASE)
    actions = apply_targets(
        test_sh, test_d, alpha_by_regime, regime_of(test_blocked)
    )
    score = v208.score_prediction(OUTPUT)
    gate_passed = bool(float(score["score_traceback_leaked"]) >= LOCAL_GATE)
    np.savez_compressed(
        ACTIONS,
        alpha=np.asarray(
            [alpha_by_regime[str(r)] for r in regime_of(test_blocked)], dtype=np.float32
        ),
        target_sh_cos_base=row_cos(test_sh, base_sh),
        target_d_cos_base=row_cos(test_d, base_d),
        **actions,
    )
    report: dict[str, object] = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A neural operator that observes neighboring official spectra and learns their "
            "position-conditioned deformation can recover propagation structure that geometry-only "
            "heads and convex KNN support transfer cannot represent."
        ),
        "compliance_boundary": {
            "generation_uses": [
                "official train H leaked S-H and PDP targets",
                "official train/test coordinates",
                "official point-cloud scene/geodesic/path features",
                "v239 strict official-data carrier",
            ],
            "generation_does_not_use": [
                "Sionna or synthetic channel labels",
                "v212/v217/v221/v222 teacher outputs",
                "synthetic replay truth during training or selection",
                "hidden labels",
                "raw RT H",
            ],
            "local_truth_usage": "one final diagnostic gate after output generation",
        },
        "parameters": {
            "neighbors": K_NEIGHBORS,
            "sh_components": SH_COMPONENTS,
            "d_components": D_COMPONENTS,
            "validation_count": int(len(val_ids)),
            "split_train_count": int(len(train_ids)),
            "valid_train_count": int(len(valid_ids)),
            "selected_epochs": selected_epochs,
            "final_seeds": list(FINAL_SEEDS),
        },
        "validation": {
            "training": validation_training,
            "metrics": validation_metrics,
            "alpha_by_regime": alpha_by_regime,
            "calibration": calibration,
        },
        "final_training": final_training,
        "test_diagnostics": {
            "target_sh_cos_base": summarize(row_cos(test_sh, base_sh)),
            "target_d_cos_base": summarize(row_cos(test_d, base_d)),
            "actions": {key: summarize(value) for key, value in actions.items()},
        },
        "local_gate": {
            "required_score_traceback_leaked": LOCAL_GATE,
            "score": score,
            "passed": gate_passed,
            "gap": float(score["score_traceback_leaked"]) - LOCAL_GATE,
        },
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "targets": PATHS.relative(TARGETS),
            "actions": PATHS.relative(ACTIONS),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    if gate_passed:
        package(report)
        report["outputs"]["zip"] = PATHS.relative(ZIP_OUT)  # type: ignore[index]
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
