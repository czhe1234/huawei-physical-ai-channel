#!/usr/bin/env python3
"""Generate v218: official-data multi-spectrum Physical-AI field.

This candidate is the compliant restart after the Sionna-backed v210-v212
experiments.  Those versions are used only as evidence that spectral shape is
the bottleneck.  This script does not read Sionna test channels, v212/v217
teachers, replay truth, restored mesh material labels, or synthetic full-H
priors.

The model is trained only from released train H plus official public geometry:

* official train H provides denoised leaked-PAS marginals
  (S-H, pol-S-H, S-H-UE, S-H-V), delay marginals (D, U-D, H-D), and power;
* official point-cloud/geodesic/path-candidate features describe each point;
* a path-token Transformer predicts multi-spectrum residual coefficients;
* deterministic projection applies the learned targets to the pre-Sionna v209
  backbone while preserving its fine phase texture and using only small,
  validation-gated power release.
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

import generate_v178_test_track_complex_consensus as track
import generate_v187_leaked_pas_lifecycle_projection as leak
import generate_v196_train_guided_delay_pdp_mouth as v196
import generate_v209_adaptive_metric_projection_controller as v209
import generate_v215_pointcloud_physics_neural_field as v215
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v218_official_multispectrum_physics_ai"
VERSION_NUMBER = 218

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v218_Round1_Test_Channel.npy"
INTERMEDIATE = OUT_DIR / "v218_after_leaked_projection.npy"
TARGETS_CACHE = OUT_DIR / "v218_train_multispectrum_targets.npz"
ACTION_OUT = OUT_DIR / "v218_official_multispectrum_actions.npz"
CHECKPOINT = PATHS.checkpoints / "v218_official_multispectrum_physics_ai.pt"
REPORT = REPORT_DIR / "v218_official_multispectrum_physics_ai_generate.json"
ZIP_OUT = PATHS.round1_versions / "v218_official_multispectrum_physics_ai_candidate_pending.zip"

BASE = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"
V215_BASE = PATHS.cache_dir(215) / "v215_Round1_Test_Channel.npy"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
TRAIN_MARGINALS = PATHS.cache_dir(196) / "v196_train_delay_marginals.npz"
TRAIN_COLL = PATHS.shared_features / "truth_coll_power.npy"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
GEODESIC_CACHE = PATHS.shared_features / "geodesic_support_features.npz"
PATH_CACHE = PATHS.shared_features / "multibounce_path_candidates.npz"

FLOOR = 1e-11
ANCHOR_K = 72
VAL_COUNT = 320
SEEDS = (21801, 21817, 21833)
TRAIN_EPOCHS = 260
FINAL_EPOCHS_MAX = 260
BATCH_SIZE = 64
TARGET_BATCH = 18
PROJ_BATCH = 5
IPF_ITERS = 6
OUTER_DELAY_ITERS = 2

PCA_DIMS = {
    "sh": 52,
    "psh": 52,
    "shue": 56,
    "shv": 56,
    "d": 32,
    "du": 42,
    "dh": 52,
}

LOSS_WEIGHTS = {
    "sh": 0.20,
    "psh": 0.10,
    "shue": 0.11,
    "shv": 0.09,
    "d": 0.24,
    "du": 0.12,
    "dh": 0.14,
}

LEAK_GAIN_CLIP = {
    "all": (0.18, 6.20),
    "top8192": (0.46, 2.55),
    "top1024": (0.60, 1.82),
    "top128": (0.76, 1.42),
}

DELAY_GAIN_CLIP = {
    "all": (0.24, 4.60),
    "top8192": (0.52, 2.20),
    "top1024": (0.66, 1.62),
    "top128": (0.82, 1.30),
}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a.reshape(len(a), -1).astype(np.float64)
    y = b.reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def l1_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(a.reshape(len(a), -1) - b.reshape(len(b), -1)), axis=1).astype(np.float32)


def denoise_distribution(x: np.ndarray, strength: float = 0.72) -> np.ndarray:
    arr = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    flat = arr.reshape(len(arr), -1)
    floor = np.quantile(flat, 0.22, axis=1).astype(np.float32)
    clean = np.maximum(flat - strength * floor[:, None], 0.0)
    empty = clean.sum(axis=1) <= 1e-20
    if np.any(empty):
        clean[empty] = flat[empty]
    return normalize(clean.reshape(arr.shape), axis=tuple(range(1, arr.ndim)))


def load_multispectrum_targets() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if TARGETS_CACHE.is_file():
        data = np.load(TARGETS_CACHE)
        targets = {key: data[key].astype(np.float32) for key in PCA_DIMS}
        return targets, data["log_power"].astype(np.float32), data["valid"].astype(bool)

    clean_delay, _raw_delay, log_power, valid_delay = v215.load_targets()
    leaked = v209.build_train_leaked_marginals()
    targets = {
        "sh": denoise_distribution(leaked["sh"], 0.74),
        "psh": denoise_distribution(leaked["psh"], 0.70),
        "shue": denoise_distribution(leaked["shue"], 0.68),
        "shv": denoise_distribution(leaked["shv"], 0.66),
        "d": clean_delay["d"],
        "du": clean_delay["du"],
        "dh": clean_delay["dh"],
    }
    valid = valid_delay.astype(bool) & leaked["valid"].astype(bool) & np.isfinite(log_power)
    np.savez_compressed(TARGETS_CACHE, **targets, log_power=log_power, valid=valid.astype(np.bool_))
    data = np.load(TARGETS_CACHE)
    return {key: data[key].astype(np.float32) for key in PCA_DIMS}, data["log_power"].astype(np.float32), data["valid"].astype(bool)


def effective_distance(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
) -> np.ndarray:
    return v215.effective_distance(query_pos, query_blocked, bank_pos, bank_blocked)


def neighbor_index_weights(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
    query_ids: np.ndarray | None = None,
    bank_ids: np.ndarray | None = None,
    k: int = ANCHOR_K,
) -> tuple[np.ndarray, np.ndarray]:
    dist = effective_distance(query_pos, query_blocked, bank_pos, bank_blocked)
    if query_ids is not None and bank_ids is not None:
        dist[query_ids[:, None] == bank_ids[None, :]] = np.inf
    k_eff = min(k, bank_pos.shape[0] - (1 if query_ids is not None else 0))
    idx = np.argpartition(dist, k_eff - 1, axis=1)[:, :k_eff]
    dsel = np.take_along_axis(dist, idx, axis=1)
    order = np.argsort(dsel, axis=1)
    idx = np.take_along_axis(idx, order, axis=1).astype(np.int32)
    dsel = np.take_along_axis(dsel, order, axis=1)
    weights = 1.0 / np.maximum(dsel, 0.35) ** 2
    same_regime = regime_of(query_blocked)[:, None] == regime_of(bank_blocked)[idx]
    weights *= np.where(same_regime, 1.22, 0.70).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    return idx, weights.astype(np.float32)


def weighted_average(target: np.ndarray, idx: np.ndarray, weights: np.ndarray, batch: int = TARGET_BATCH) -> np.ndarray:
    out = np.empty((len(idx),) + target.shape[1:], dtype=np.float32)
    for start in range(0, len(idx), batch):
        end = min(start + batch, len(idx))
        selected = target[idx[start:end]]
        shape = (end - start, idx.shape[1]) + (1,) * (target.ndim - 1)
        out[start:end] = np.sum(selected * weights[start:end].reshape(shape), axis=1).astype(np.float32)
        print(f"v218 anchor average {target.shape[1:]} {end}/{len(idx)}", flush=True)
    return normalize(out, axis=tuple(range(1, out.ndim)))


@dataclass
class Prepared:
    global_features: np.ndarray
    path_numeric: np.ndarray
    path_type: np.ndarray
    path_valid: np.ndarray
    anchors: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    coeff: dict[str, np.ndarray]
    anchor_log_power: np.ndarray
    target_log_power: np.ndarray
    regimes: np.ndarray


def fit_basis(anchor: np.ndarray, target: np.ndarray, components: int) -> tuple[v215.Basis, np.ndarray]:
    basis, coeff = v215.fit_basis(anchor, target, components)
    return basis, coeff


def make_coeff(anchor: np.ndarray, target: np.ndarray, basis: v215.Basis) -> np.ndarray:
    residual = np.clip(
        np.log(np.maximum(target.reshape(len(target), -1), FLOOR))
        - np.log(np.maximum(anchor.reshape(len(anchor), -1), FLOOR)),
        -8.0,
        8.0,
    )
    raw = (residual - basis.mean[None, :]) @ basis.components.T
    return (raw / basis.coeff_scale[None, :]).astype(np.float32)


def build_prepared(
    prefix: str,
    indices: np.ndarray | None,
    pos: np.ndarray,
    blocked: np.ndarray,
    targets: dict[str, np.ndarray],
    log_power: np.ndarray,
    anchors: dict[str, np.ndarray],
    anchor_log_power: np.ndarray,
    bases: dict[str, v215.Basis],
    feature_stats: dict[str, np.ndarray] | None,
) -> tuple[Prepared, dict[str, np.ndarray]]:
    delay_anchor = v215.Anchors(
        beam=anchors["sh"].sum(axis=1),
        d=anchors["d"],
        du=anchors["du"],
        dh=anchors["dh"],
        log_power=anchor_log_power.astype(np.float32),
        agreement=np.ones(len(pos), dtype=np.float32),
        effective_count=np.full(len(pos), float(ANCHOR_K), dtype=np.float32),
    )
    global_raw = v215.make_global_features(prefix, indices, pos, blocked, delay_anchor)
    path_raw, path_type, path_valid = v215.make_path_tokens(prefix, pos, indices)
    if feature_stats is None:
        valid_path = path_raw[path_valid]
        feature_stats = {
            "global_mean": global_raw.mean(axis=0).astype(np.float32),
            "global_std": np.maximum(global_raw.std(axis=0), 1e-5).astype(np.float32),
            "path_mean": valid_path.mean(axis=0).astype(np.float32),
            "path_std": np.maximum(valid_path.std(axis=0), 1e-5).astype(np.float32),
        }
    global_features = np.clip(
        (global_raw - feature_stats["global_mean"]) / feature_stats["global_std"],
        -8.0,
        8.0,
    ).astype(np.float32)
    path_numeric = np.clip(
        (path_raw - feature_stats["path_mean"][None, None, :]) / feature_stats["path_std"][None, None, :],
        -8.0,
        8.0,
    ).astype(np.float32)
    path_numeric[~path_valid] = 0.0
    coeff = {key: make_coeff(anchors[key], targets[key], bases[key]) for key in PCA_DIMS}
    return (
        Prepared(
            global_features=global_features,
            path_numeric=path_numeric,
            path_type=path_type,
            path_valid=path_valid,
            anchors=anchors,
            targets=targets,
            coeff=coeff,
            anchor_log_power=anchor_log_power.astype(np.float32),
            target_log_power=log_power.astype(np.float32),
            regimes=regime_of(blocked),
        ),
        feature_stats,
    )


class MultiSpectrumField(nn.Module):
    def __init__(self, global_dim: int, path_dim: int, output_dims: dict[str, int]):
        super().__init__()
        d_model = 192
        self.type_emb = nn.Embedding(4, 24)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, 384),
            nn.SiLU(),
            nn.LayerNorm(384),
            nn.Dropout(0.05),
            nn.Linear(384, d_model),
        )
        self.path_proj = nn.Sequential(
            nn.Linear(path_dim + 24, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=6,
            dim_feedforward=640,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 384),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
        )
        self.heads = nn.ModuleDict({key: nn.Linear(256, dim) for key, dim in output_dims.items()})
        self.power_head = nn.Linear(256, 1)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        query = self.global_proj(global_features).unsqueeze(1)
        path = self.path_proj(torch.cat([path_numeric, self.type_emb(path_type)], dim=2))
        tokens = torch.cat([query, path], dim=1)
        mask = torch.cat(
            [torch.zeros((len(path_valid), 1), dtype=torch.bool, device=path_valid.device), ~path_valid],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        latent = self.trunk(encoded[:, 0])
        return {key: head(latent) for key, head in self.heads.items()}, self.power_head(latent).squeeze(1)


def torch_basis(basis: v215.Basis, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(basis.mean).to(device),
        torch.from_numpy(basis.components).to(device),
        torch.from_numpy(basis.coeff_scale).to(device),
    )


def reconstruct_shape(
    coeff_norm: torch.Tensor,
    anchor: torch.Tensor,
    basis_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    mean, components, scale = basis_tensors
    delta = (coeff_norm * scale[None, :]) @ components + mean[None, :]
    logits = torch.log(anchor.reshape(len(anchor), -1).clamp_min(FLOOR)) + delta
    return torch.softmax(logits, dim=1)


def train_model(
    prepared: Prepared,
    bases: dict[str, v215.Basis],
    seed: int,
    epochs: int,
    device: torch.device,
    validation: Prepared | None = None,
) -> tuple[MultiSpectrumField, dict[str, object]]:
    set_seed(seed)
    model = MultiSpectrumField(
        prepared.global_features.shape[1],
        prepared.path_numeric.shape[2],
        {key: len(bases[key].coeff_scale) for key in bases},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.5e-4, weight_decay=2.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    basis_gpu = {key: torch_basis(value, device) for key, value in bases.items()}
    rng = np.random.default_rng(seed)
    n = len(prepared.global_features)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses_epoch = []
        for start in range(0, n, BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            coeff_pred, power_delta = model(
                torch.from_numpy(prepared.global_features[ids]).to(device),
                torch.from_numpy(prepared.path_numeric[ids]).to(device),
                torch.from_numpy(prepared.path_type[ids]).to(device),
                torch.from_numpy(prepared.path_valid[ids]).to(device),
            )
            loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            coeff_loss_total = torch.tensor(0.0, dtype=torch.float32, device=device)
            for key, weight in LOSS_WEIGHTS.items():
                anchor = torch.from_numpy(prepared.anchors[key][ids]).to(device)
                truth = torch.from_numpy(prepared.targets[key][ids]).to(device)
                coeff_truth = torch.from_numpy(prepared.coeff[key][ids]).to(device)
                pred = reconstruct_shape(coeff_pred[key], anchor, basis_gpu[key])
                cos_loss = 1.0 - F.cosine_similarity(pred, truth.reshape(len(ids), -1), dim=1).mean()
                coeff_loss = F.smooth_l1_loss(coeff_pred[key], coeff_truth, beta=0.55)
                loss = loss + weight * (0.76 * cos_loss + 0.24 * coeff_loss)
                coeff_loss_total = coeff_loss_total + coeff_loss
            target_power_delta = torch.from_numpy(
                prepared.target_log_power[ids] - prepared.anchor_log_power[ids]
            ).to(device)
            power_loss = F.smooth_l1_loss(power_delta, target_power_delta, beta=0.75)
            loss = loss + 0.055 * power_loss + 0.015 * coeff_loss_total / len(PCA_DIMS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses_epoch.append(float(loss.detach().cpu()))
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item = {"epoch": float(epoch), "train_loss": float(np.mean(losses_epoch))}
            if validation is not None:
                pred = predict_model(model, validation, bases, device)
                support_score = 0.0
                anchor_score = 0.0
                for key, weight in LOSS_WEIGHTS.items():
                    support_score += weight * float(np.mean(row_cos(pred[key], validation.targets[key])))
                    anchor_score += weight * float(np.mean(row_cos(validation.anchors[key], validation.targets[key])))
                calibrated = []
                alpha_grid = np.linspace(0.0, 1.20, 25, dtype=np.float32)
                for alpha in alpha_grid:
                    score = 0.0
                    for key, weight in LOSS_WEIGHTS.items():
                        axes = tuple(range(1, validation.anchors[key].ndim))
                        blend = normalize((1.0 - alpha) * validation.anchors[key] + alpha * pred[key], axis=axes)
                        score += weight * float(np.mean(row_cos(blend, validation.targets[key])))
                    calibrated.append(score)
                item["val_anchor_score"] = anchor_score
                item["val_model_score"] = support_score
                item["val_calibrated_score"] = float(np.max(calibrated))
                item["val_alpha"] = float(alpha_grid[int(np.argmax(calibrated))])
                if item["val_calibrated_score"] > best_metric:
                    best_metric = item["val_calibrated_score"]
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v218 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_calibrated_score": None if not np.isfinite(best_metric) else float(best_metric),
    }


@torch.no_grad()
def predict_model(
    model: MultiSpectrumField,
    prepared: Prepared,
    bases: dict[str, v215.Basis],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    basis_gpu = {key: torch_basis(value, device) for key, value in bases.items()}
    out: dict[str, list[np.ndarray]] = {key: [] for key in bases}
    power = []
    for start in range(0, len(prepared.global_features), 192):
        end = min(start + 192, len(prepared.global_features))
        coeff, power_delta = model(
            torch.from_numpy(prepared.global_features[start:end]).to(device),
            torch.from_numpy(prepared.path_numeric[start:end]).to(device),
            torch.from_numpy(prepared.path_type[start:end]).to(device),
            torch.from_numpy(prepared.path_valid[start:end]).to(device),
        )
        for key, basis in bases.items():
            anchor = torch.from_numpy(prepared.anchors[key][start:end]).to(device)
            shape = reconstruct_shape(coeff[key], anchor, basis_gpu[key])
            out[key].append(shape.cpu().numpy().reshape((end - start,) + basis.shape).astype(np.float32))
        power.append((power_delta.cpu().numpy() + prepared.anchor_log_power[start:end]).astype(np.float32))
    result = {key: np.concatenate(value).astype(np.float32) for key, value in out.items()}
    result["log_power"] = np.concatenate(power).astype(np.float32)
    return result


def calibration(
    pred: dict[str, np.ndarray],
    val: Prepared,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, object]]:
    alpha_grid = np.linspace(0.0, 1.20, 25, dtype=np.float32)
    support_alpha: dict[str, float] = {}
    leak_cap: dict[str, float] = {}
    delay_cap: dict[str, float] = {}
    power_alpha: dict[str, float] = {}
    report: dict[str, object] = {}
    for regime in ("clear", "light", "medium", "heavy"):
        mask = val.regimes.astype(str) == regime
        if not np.any(mask):
            support_alpha[regime] = 0.0
            leak_cap[regime] = 0.0
            delay_cap[regime] = 0.0
            power_alpha[regime] = 0.0
            continue
        scores = []
        for alpha in alpha_grid:
            score = 0.0
            for key, weight in LOSS_WEIGHTS.items():
                axes = tuple(range(1, val.anchors[key].ndim))
                blend = normalize((1.0 - alpha) * val.anchors[key][mask] + alpha * pred[key][mask], axis=axes)
                score += weight * float(np.mean(row_cos(blend, val.targets[key][mask])))
            scores.append(score)
        best = int(np.argmax(scores))
        raw_alpha = float(alpha_grid[best])
        cap_by_regime = {"clear": 0.0, "light": 0.16, "medium": 0.74, "heavy": 0.92}
        deployed_alpha = min(raw_alpha, cap_by_regime[regime])
        support_alpha[regime] = deployed_alpha
        sh_action = np.sum(
            np.abs(
                normalize(
                    (1.0 - deployed_alpha) * val.anchors["sh"][mask]
                    + deployed_alpha * pred["sh"][mask],
                    axis=(1, 2),
                )
                - val.anchors["sh"][mask]
            ),
            axis=(1, 2),
        )
        d_action = np.sum(
            np.abs(
                normalize(
                    (1.0 - deployed_alpha) * val.anchors["d"][mask]
                    + deployed_alpha * pred["d"][mask],
                    axis=1,
                )
                - val.anchors["d"][mask]
            ),
            axis=1,
        )
        leak_cap[regime] = float(np.quantile(sh_action, 0.88))
        delay_cap[regime] = float(np.quantile(d_action, 0.88))

        p_scores = []
        for alpha in alpha_grid:
            value = val.anchor_log_power[mask] + alpha * (pred["log_power"][mask] - val.anchor_log_power[mask])
            p_scores.append(float(np.mean(np.abs(value - val.target_log_power[mask]))))
        p_best = int(np.argmin(p_scores))
        if p_scores[p_best] + 0.025 < p_scores[0]:
            power_alpha[regime] = min(float(alpha_grid[p_best]), {"clear": 0.08, "light": 0.08, "medium": 0.16, "heavy": 0.16}[regime])
        else:
            power_alpha[regime] = 0.0
        report[regime] = {
            "count": int(np.sum(mask)),
            "support_alpha_raw": raw_alpha,
            "support_alpha_deployed": deployed_alpha,
            "support_score_anchor": float(scores[0]),
            "support_score_best": float(scores[best]),
            "leak_sh_action_p88": leak_cap[regime],
            "delay_d_action_p88": delay_cap[regime],
            "power_alpha_deployed": power_alpha[regime],
            "power_mae_anchor": float(p_scores[0]),
            "power_mae_best": float(p_scores[p_best]),
        }
    return support_alpha, leak_cap, delay_cap, power_alpha, report


def build_data() -> tuple[Prepared, Prepared, Prepared, dict[str, v215.Basis], dict[str, v215.Basis], dict[str, object]]:
    targets_all, log_power_all, valid = load_multispectrum_targets()
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    valid_ids = np.flatnonzero(valid)
    matched = choose_val_indices(train_pos[:, :2], test_pos[:, :2], VAL_COUNT, "matched-test", VERSION_NUMBER)
    val_ids = np.intersect1d(matched, valid_ids)
    train_ids = np.setdiff1d(valid_ids, val_ids)

    idx_train, w_train = neighbor_index_weights(
        train_pos[train_ids],
        train_blocked[train_ids],
        train_pos[train_ids],
        train_blocked[train_ids],
        query_ids=train_ids,
        bank_ids=train_ids,
    )
    idx_val, w_val = neighbor_index_weights(
        train_pos[val_ids],
        train_blocked[val_ids],
        train_pos[train_ids],
        train_blocked[train_ids],
    )
    train_targets = {key: targets_all[key][train_ids] for key in PCA_DIMS}
    val_targets = {key: targets_all[key][val_ids] for key in PCA_DIMS}
    train_anchor = {key: weighted_average(train_targets[key], idx_train, w_train) for key in PCA_DIMS}
    val_anchor = {key: weighted_average(train_targets[key], idx_val, w_val) for key in PCA_DIMS}
    train_anchor_power = np.sum(log_power_all[train_ids][idx_train] * w_train, axis=1).astype(np.float32)
    val_anchor_power = np.sum(log_power_all[train_ids][idx_val] * w_val, axis=1).astype(np.float32)

    bases: dict[str, v215.Basis] = {}
    for key, dims in PCA_DIMS.items():
        basis, _coeff = fit_basis(train_anchor[key], train_targets[key], dims)
        bases[key] = basis
        print(f"v218 fit basis {key} dims={len(basis.coeff_scale)}", flush=True)

    train_prepared, feature_stats = build_prepared(
        "train",
        train_ids,
        train_pos[train_ids],
        train_blocked[train_ids],
        train_targets,
        log_power_all[train_ids],
        train_anchor,
        train_anchor_power,
        bases,
        None,
    )
    val_prepared, _ = build_prepared(
        "train",
        val_ids,
        train_pos[val_ids],
        train_blocked[val_ids],
        val_targets,
        log_power_all[val_ids],
        val_anchor,
        val_anchor_power,
        bases,
        feature_stats,
    )

    idx_final, w_final = neighbor_index_weights(
        train_pos[valid_ids],
        train_blocked[valid_ids],
        train_pos[valid_ids],
        train_blocked[valid_ids],
        query_ids=valid_ids,
        bank_ids=valid_ids,
    )
    final_targets = {key: targets_all[key][valid_ids] for key in PCA_DIMS}
    final_anchor = {key: weighted_average(final_targets[key], idx_final, w_final) for key in PCA_DIMS}
    final_anchor_power = np.sum(log_power_all[valid_ids][idx_final] * w_final, axis=1).astype(np.float32)
    final_bases: dict[str, v215.Basis] = {}
    for key, dims in PCA_DIMS.items():
        basis, _coeff = fit_basis(final_anchor[key], final_targets[key], dims)
        final_bases[key] = basis
        print(f"v218 fit final basis {key} dims={len(basis.coeff_scale)}", flush=True)

    final_prepared, final_feature_stats = build_prepared(
        "train",
        valid_ids,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        final_targets,
        log_power_all[valid_ids],
        final_anchor,
        final_anchor_power,
        final_bases,
        None,
    )
    idx_test, w_test = neighbor_index_weights(
        test_pos,
        test_blocked,
        train_pos[valid_ids],
        train_blocked[valid_ids],
    )
    test_anchor = {key: weighted_average(final_targets[key], idx_test, w_test) for key in PCA_DIMS}
    test_anchor_power = np.sum(log_power_all[valid_ids][idx_test] * w_test, axis=1).astype(np.float32)
    dummy_targets = {key: test_anchor[key].copy() for key in PCA_DIMS}
    test_prepared, _ = build_prepared(
        "test",
        None,
        test_pos,
        test_blocked,
        dummy_targets,
        test_anchor_power,
        test_anchor,
        test_anchor_power,
        final_bases,
        final_feature_stats,
    )
    diagnostics = {
        "valid_train_count": int(len(valid_ids)),
        "train_count": int(len(train_ids)),
        "validation_count": int(len(val_ids)),
        "feature_dim": int(train_prepared.global_features.shape[1]),
        "path_dim": int(train_prepared.path_numeric.shape[2]),
        "target_shapes": {key: list(value.shape[1:]) for key, value in targets_all.items()},
        "train_anchor_scores": {
            key: summarize(row_cos(final_anchor[key], final_targets[key])) for key in PCA_DIMS
        },
        "regimes_test": test_prepared.regimes,
        "final_prepared": final_prepared,
        "valid_ids": valid_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "feature_stats": final_feature_stats,
        "test_neighbor_idx": idx_test.astype(np.int32),
        "test_neighbor_weights": w_test.astype(np.float32),
    }
    return train_prepared, val_prepared, test_prepared, bases, final_bases, diagnostics


def blend_low_marginals(
    current: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    regimes: np.ndarray,
    support_alpha: dict[str, float],
    leak_cap: dict[str, float],
    delay_cap: dict[str, float],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    base_alpha = np.asarray([support_alpha[str(r)] for r in regimes], dtype=np.float32)
    leak_alpha = base_alpha.copy()
    delay_alpha = base_alpha.copy()
    leak_action = l1_rows(prediction["sh"], current["sh"])
    delay_action = l1_rows(prediction["d"], current["d"])
    leak_limits = np.asarray([leak_cap[str(r)] for r in regimes], dtype=np.float32)
    delay_limits = np.asarray([delay_cap[str(r)] for r in regimes], dtype=np.float32)
    leak_alpha *= np.minimum(1.0, leak_limits / np.maximum(base_alpha * leak_action, 1e-8)).astype(np.float32)
    delay_alpha *= np.minimum(1.0, delay_limits / np.maximum(base_alpha * delay_action, 1e-8)).astype(np.float32)

    desired: dict[str, np.ndarray] = {}
    for key in ("sh", "psh", "shue", "shv"):
        axes = tuple(range(1, current[key].ndim))
        shape = (len(regimes),) + (1,) * (current[key].ndim - 1)
        desired[key] = normalize(
            (1.0 - leak_alpha.reshape(shape)) * current[key]
            + leak_alpha.reshape(shape) * prediction[key],
            axis=axes,
        )
    for key in ("d", "du", "dh"):
        axes = tuple(range(1, current[key].ndim))
        shape = (len(regimes),) + (1,) * (current[key].ndim - 1)
        desired[key] = normalize(
            (1.0 - delay_alpha.reshape(shape)) * current[key]
            + delay_alpha.reshape(shape) * prediction[key],
            axis=axes,
        )
    return desired, leak_alpha, delay_alpha


def support_marginals(full_support: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "sh": normalize(np.asarray(full_support).sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32), axis=(1, 2)),
        "psh": normalize(np.asarray(full_support).sum(axis=(4, 5), dtype=np.float64).astype(np.float32), axis=(1, 2, 3)),
        "shue": normalize(np.asarray(full_support).sum(axis=(1, 5), dtype=np.float64).astype(np.float32), axis=(1, 2, 3)),
        "shv": normalize(np.asarray(full_support).sum(axis=(1, 4), dtype=np.float64).astype(np.float32), axis=(1, 2, 3)),
    }


def apply_leaked_targets(targets: dict[str, np.ndarray], regimes: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(INTERMEDIATE, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "leaked_full_l1_actual": [],
        "leaked_sh_l1_actual": [],
        "leaked_power_factor": [],
        "leaked_gain_min": [],
        "leaked_gain_p10": [],
        "leaked_gain_p90": [],
        "leaked_gain_max": [],
    }
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        spec0 = leak.leaked_forward(h0)
        power0 = leak.leaked_cell_power(spec0)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        work = support0.copy()
        for _ in range(IPF_ITERS):
            v209.apply_leaked_ratio(
                work,
                safe_ratio(targets["sh"][start:end], work.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)),
                "sh",
                0.82,
            )
            v209.apply_leaked_ratio(
                work,
                safe_ratio(targets["psh"][start:end], work.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)),
                "psh",
                0.48,
            )
            v209.apply_leaked_ratio(
                work,
                safe_ratio(targets["shue"][start:end], work.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)),
                "shue",
                0.52,
            )
            v209.apply_leaked_ratio(
                work,
                safe_ratio(targets["shv"][start:end], work.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)),
                "shv",
                0.46,
            )
            work = normalize(work.reshape(end - start, -1), axis=1).reshape(support0.shape)
        target_power = work * total0[:, None, None, None, None, None]
        gain = np.sqrt(target_power / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, *LEAK_GAIN_CLIP["all"]).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], *LEAK_GAIN_CLIP["top8192"])
        gain[strong1024] = np.clip(gain[strong1024], *LEAK_GAIN_CLIP["top1024"])
        gain[strong128] = np.clip(gain[strong128], *LEAK_GAIN_CLIP["top128"])
        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.42 * (gain[local] - 1.0)
        h1 = leak.leaked_inverse(spec0 * gain)
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite leaked block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()
        power1 = leak.leaked_cell_power(leak.leaked_forward(h1.astype(np.complex64)))
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        diag["leaked_full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["leaked_sh_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 4, 5), dtype=np.float64)
                    - support0.sum(axis=(1, 4, 5), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        diag["leaked_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["leaked_gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["leaked_gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["leaked_gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["leaked_gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v218 leaked projection {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def delay_marginals_from_support(support: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "d": normalize(support.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32), axis=1),
        "du": normalize(support.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32), axis=(1, 2)),
        "dh": normalize(support.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32), axis=(1, 2)),
    }


def apply_delay_and_power(
    targets: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    regimes: np.ndarray,
    power_alpha: dict[str, float],
) -> dict[str, np.ndarray]:
    source = np.load(INTERMEDIATE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    alpha_p = np.asarray([power_alpha[str(r)] for r in regimes], dtype=np.float32)
    diag: dict[str, list[np.ndarray]] = {
        "delay_full_l1_actual": [],
        "delay_d_l1_actual": [],
        "delay_power_factor": [],
        "final_power_factor": [],
        "extra_log_delta": [],
        "delay_gain_min": [],
        "delay_gain_p10": [],
        "delay_gain_p90": [],
        "delay_gain_max": [],
    }
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        work_h = h0.copy()
        for _ in range(OUTER_DELAY_ITERS):
            cir = np.fft.ifft(work_h, axis=-1)
            arr = cir.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
            power0 = (np.abs(arr) ** 2).astype(np.float32)
            total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
            support = power0 / total0[:, None, None, None, None, None]
            for _ in range(IPF_ITERS):
                v196.apply_delay_ratio(
                    support,
                    safe_ratio(targets["d"][start:end], support.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)),
                    "d",
                    0.98,
                )
                v196.apply_delay_ratio(
                    support,
                    safe_ratio(targets["du"][start:end], support.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)),
                    "du",
                    0.36,
                )
                v196.apply_delay_ratio(
                    support,
                    safe_ratio(targets["dh"][start:end], support.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)),
                    "dh",
                    0.34,
                )
                support = normalize(support.reshape(end - start, -1), axis=1).reshape(power0.shape)
            gain = np.sqrt((support * total0[:, None, None, None, None, None]) / np.maximum(power0, 1e-30))
            gain = np.clip(gain, *DELAY_GAIN_CLIP["all"]).astype(np.float32)
            flat = (power0 / np.maximum(power0.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)).reshape(end - start, -1)
            strong128 = top_mask(flat, 128).reshape(power0.shape)
            strong1024 = top_mask(flat, 1024).reshape(power0.shape)
            strong8192 = top_mask(flat, 8192).reshape(power0.shape)
            gain[strong8192] = np.clip(gain[strong8192], *DELAY_GAIN_CLIP["top8192"])
            gain[strong1024] = np.clip(gain[strong1024], *DELAY_GAIN_CLIP["top1024"])
            gain[strong128] = np.clip(gain[strong128], *DELAY_GAIN_CLIP["top128"])
            for local, regime in enumerate(regimes[start:end].astype(str)):
                if regime == "clear":
                    gain[local] = 1.0
                elif regime == "light":
                    gain[local] = 1.0 + 0.30 * (gain[local] - 1.0)
            work_h = np.fft.fft((arr * gain).reshape(end - start, 256, N_ANT, S_SC), axis=-1).astype(np.complex64)

        h_delay_power = np.maximum(np.sum(np.abs(work_h) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        current_delay_total = np.maximum(v196.delay_power_from_h(work_h).sum(axis=(1, 2, 3, 4, 5)), TINY)
        current_log = np.log(current_delay_total).astype(np.float32)
        raw_delta = alpha_p[start:end] * (prediction["log_power"][start:end] - current_log)
        log_delta = np.clip(raw_delta, -0.055, 0.055).astype(np.float32)
        work_h *= np.exp(0.5 * log_delta)[:, None, None, None].astype(np.float32)
        if not (np.isfinite(work_h.real).all() and np.isfinite(work_h.imag).all()):
            raise FloatingPointError(f"non-finite delay block {start}:{end}")
        output[start:end] = work_h.astype(np.complex64)
        output.flush()

        final_power = v196.delay_power_from_h(work_h.astype(np.complex64))
        final_support = final_power / np.maximum(final_power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        base_power = v196.delay_power_from_h(h0.astype(np.complex64))
        base_support = base_power / np.maximum(base_power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        final_m = delay_marginals_from_support(final_support)
        base_m = delay_marginals_from_support(base_support)
        diag["delay_full_l1_actual"].append(np.sum(np.abs(final_support - base_support), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["delay_d_l1_actual"].append(np.sum(np.abs(final_m["d"] - base_m["d"]), axis=1).astype(np.float32))
        diag["delay_power_factor"].append((h_delay_power / h_power0).astype(np.float32))
        diag["final_power_factor"].append(
            (np.sum(np.abs(work_h) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diag["extra_log_delta"].append(log_delta)
        gflat = gain.reshape(end - start, -1)
        diag["delay_gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["delay_gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["delay_gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["delay_gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v218 delay/power projection {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def generate() -> dict[str, object]:
    for path in [BASE, GEOM_CACHE, TRAIN_MARGINALS, TRAIN_COLL, SCENE_CACHE, GEODESIC_CACHE, PATH_CACHE]:
        require_file(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_prepared, val_prepared, test_prepared, validation_bases, final_bases, data_diag = build_data()

    validation_model, validation_history = train_model(
        train_prepared, validation_bases, SEEDS[0], TRAIN_EPOCHS, device, validation=val_prepared
    )
    val_pred = predict_model(validation_model, val_prepared, validation_bases, device)
    support_alpha, leak_cap, delay_cap, power_alpha, cal_report = calibration(val_pred, val_prepared)
    validation_diag = {
        "calibration": cal_report,
        "support_alpha": support_alpha,
        "leak_cap": leak_cap,
        "delay_cap": delay_cap,
        "power_alpha": power_alpha,
        "scores": {
            key: {
                "anchor": summarize(row_cos(val_prepared.anchors[key], val_prepared.targets[key])),
                "model": summarize(row_cos(val_pred[key], val_prepared.targets[key])),
            }
            for key in PCA_DIMS
        },
        "log_power_mae_anchor": float(np.mean(np.abs(val_prepared.anchor_log_power - val_prepared.target_log_power))),
        "log_power_mae_model": float(np.mean(np.abs(val_pred["log_power"] - val_prepared.target_log_power))),
    }
    selected_epochs = max(1, min(FINAL_EPOCHS_MAX, int(validation_history["best_epoch"])))
    del validation_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_prepared: Prepared = data_diag["final_prepared"]
    test_predictions = []
    state_dicts = []
    final_histories = []
    for seed in SEEDS:
        model, history = train_model(final_prepared, final_bases, seed, selected_epochs, device)
        test_predictions.append(predict_model(model, test_prepared, final_bases, device))
        state_dicts.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        final_histories.append(history)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble = {
        key: np.mean(np.stack([pred[key] for pred in test_predictions], axis=0), axis=0).astype(np.float32)
        for key in (*PCA_DIMS.keys(), "log_power")
    }

    current_support = v209.build_current_leaked_support()
    current_low = support_marginals(current_support)
    source = np.load(BASE, mmap_mode="r")
    current_delay = []
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        power = v196.delay_power_from_h(np.asarray(source[start:end], dtype=np.complex64))
        support = power / np.maximum(power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        current_delay.append(delay_marginals_from_support(support))
    current_delay_m = {
        key: normalize(np.concatenate([part[key] for part in current_delay], axis=0), axis=tuple(range(1, current_delay[0][key].ndim)))
        for key in ("d", "du", "dh")
    }
    current = {**current_low, **current_delay_m}
    desired, leak_alpha, delay_alpha = blend_low_marginals(
        current,
        ensemble,
        data_diag["regimes_test"],
        support_alpha,
        leak_cap,
        delay_cap,
    )
    leaked_diag = apply_leaked_targets(desired, data_diag["regimes_test"])
    final_diag = apply_delay_and_power(desired, ensemble, data_diag["regimes_test"], power_alpha)

    np.savez_compressed(
        ACTION_OUT,
        regimes=data_diag["regimes_test"].astype("U8"),
        leak_alpha=leak_alpha,
        delay_alpha=delay_alpha,
        support_alpha_by_regime=np.asarray([support_alpha[str(r)] for r in data_diag["regimes_test"]], dtype=np.float32),
        power_alpha_by_regime=np.asarray([power_alpha[str(r)] for r in data_diag["regimes_test"]], dtype=np.float32),
        desired_sh=desired["sh"],
        desired_d=desired["d"],
        desired_du=desired["du"],
        desired_dh=desired["dh"],
        desired_log_power=ensemble["log_power"],
        test_neighbor_idx=data_diag["test_neighbor_idx"],
        test_neighbor_weights=data_diag["test_neighbor_weights"],
        **leaked_diag,
        **final_diag,
    )

    checkpoint_payload = {
        "method": VERSION,
        "seeds": SEEDS,
        "selected_epochs": selected_epochs,
        "state_dicts": state_dicts,
        "global_dim": final_prepared.global_features.shape[1],
        "path_dim": final_prepared.path_numeric.shape[2],
        "pca_dims": PCA_DIMS,
        "feature_stats": data_diag["feature_stats"],
        "bases": {
            key: {
                "mean": basis.mean,
                "components": basis.components,
                "coeff_scale": basis.coeff_scale,
                "shape": basis.shape,
            }
            for key, basis in final_bases.items()
        },
        "support_alpha": support_alpha,
        "leak_cap": leak_cap,
        "delay_cap": delay_cap,
        "power_alpha": power_alpha,
        "official_inputs_only": True,
        "no_sionna_test_teacher": True,
        "backbone": PATHS.relative(BASE),
    }
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, CHECKPOINT)

    comparison_to_v209 = track.compare_output_to(OUTPUT, BASE, data_diag["regimes_test"])
    comparison_to_v215 = track.compare_output_to(OUTPUT, V215_BASE, data_diag["regimes_test"]) if V215_BASE.is_file() else {}
    if comparison_to_v209["shape"] != [500, 256, 4, 192] or comparison_to_v209["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v209['shape']} {comparison_to_v209['dtype']}")
    if not comparison_to_v209["finite"]:
        raise ValueError("v218 output contains non-finite values")

    action_diag = {**leaked_diag, **final_diag}
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v215 failed because the neural target was too narrow.  v218 trains "
            "one official-data model on the evaluator-sensitive leaked PAS "
            "marginals, delay/PDP marginals and power confidence together, then "
            "projects only bounded multi-spectrum corrections onto v209."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H",
                "official train/test coordinates",
                "official point-cloud-derived scene/geodesic/path tokens",
                "pre-Sionna v209 AI backbone",
            ],
            "does_not_use": [
                "Sionna test full-H target",
                "v212/v217 teacher outputs",
                "synthetic replay clean truth",
                "raw RT phase or total power",
                "hidden test labels",
            ],
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "seeds": SEEDS,
            "train_epochs": TRAIN_EPOCHS,
            "selected_final_epochs": selected_epochs,
            "pca_dims": PCA_DIMS,
            "loss_weights": LOSS_WEIGHTS,
            "anchor_k": ANCHOR_K,
            "ipf_iters": IPF_ITERS,
            "outer_delay_iters": OUTER_DELAY_ITERS,
            "projection_order": ["leaked PAS multi-marginal", "delay/PDP multi-marginal", "validation-gated tiny power"],
            "phase_policy": "preserve v209 local phase texture; no absolute phase regression or phase teacher",
        },
        "data_diagnostics": {
            "valid_train_count": data_diag["valid_train_count"],
            "train_count": data_diag["train_count"],
            "validation_count": data_diag["validation_count"],
            "feature_dim": data_diag["feature_dim"],
            "path_dim": data_diag["path_dim"],
            "target_shapes": data_diag["target_shapes"],
            "train_anchor_scores": data_diag["train_anchor_scores"],
        },
        "validation": validation_diag,
        "training": {
            "validation_history": validation_history,
            "final_histories": final_histories,
        },
        "action_counts": {
            "leak_active": int(np.sum(leak_alpha > 1e-7)),
            "delay_active": int(np.sum(delay_alpha > 1e-7)),
            "power_active": int(np.sum(np.abs(final_diag["extra_log_delta"]) > 1e-8)),
            "by_regime": {
                regime: {
                    "count": int(np.sum(data_diag["regimes_test"].astype(str) == regime)),
                    "leak_active": int(np.sum((data_diag["regimes_test"].astype(str) == regime) & (leak_alpha > 1e-7))),
                    "delay_active": int(np.sum((data_diag["regimes_test"].astype(str) == regime) & (delay_alpha > 1e-7))),
                    "power_active": int(
                        np.sum(
                            (data_diag["regimes_test"].astype(str) == regime)
                            & (np.abs(final_diag["extra_log_delta"]) > 1e-8)
                        )
                    ),
                }
                for regime in ("clear", "light", "medium", "heavy")
            },
        },
        "diagnostics": {
            "overall": {key: summarize(value) for key, value in action_diag.items()},
            "by_regime": by_regime_summary(
                data_diag["regimes_test"],
                {
                    "leak_alpha": leak_alpha,
                    "delay_alpha": delay_alpha,
                    "leaked_full_l1_actual": leaked_diag["leaked_full_l1_actual"],
                    "leaked_sh_l1_actual": leaked_diag["leaked_sh_l1_actual"],
                    "delay_full_l1_actual": final_diag["delay_full_l1_actual"],
                    "delay_d_l1_actual": final_diag["delay_d_l1_actual"],
                    "extra_log_delta": final_diag["extra_log_delta"],
                    "final_power_factor": final_diag["final_power_factor"],
                },
            ),
        },
        "comparison_to_v209": comparison_to_v209,
        "comparison_to_v215": comparison_to_v215,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "intermediate": PATHS.relative(INTERMEDIATE),
            "targets_cache": PATHS.relative(TARGETS_CACHE),
            "actions": PATHS.relative(ACTION_OUT),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def finalize_existing_report() -> dict[str, object]:
    actions = np.load(ACTION_OUT)
    regimes = actions["regimes"].astype(str)
    action_diag = {
        key: actions[key].astype(np.float32)
        for key in actions.files
        if key
        not in {
            "regimes",
            "desired_sh",
            "desired_d",
            "desired_du",
            "desired_dh",
            "desired_log_power",
            "test_neighbor_idx",
            "test_neighbor_weights",
        }
        and np.issubdtype(actions[key].dtype, np.number)
        and actions[key].ndim <= 1
    }
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    comparison_to_v209 = track.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v215 = track.compare_output_to(OUTPUT, V215_BASE, regimes) if V215_BASE.is_file() else {}
    if comparison_to_v209["shape"] != [500, 256, 4, 192] or comparison_to_v209["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v209['shape']} {comparison_to_v209['dtype']}")
    if not comparison_to_v209["finite"]:
        raise ValueError("v218 output contains non-finite values")
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "resume_note": "Finalized from existing output/action/checkpoint after report JSON serialization failed.",
        "primary_hypothesis": (
            "v218 trains one official-data model on evaluator-sensitive leaked PAS "
            "marginals, delay/PDP marginals and power confidence together, then "
            "projects bounded multi-spectrum corrections onto v209."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H",
                "official train/test coordinates",
                "official point-cloud-derived scene/geodesic/path tokens",
                "pre-Sionna v209 AI backbone",
            ],
            "does_not_use": [
                "Sionna test full-H target",
                "v212/v217 teacher outputs",
                "synthetic replay clean truth",
                "raw RT phase or total power",
                "hidden test labels",
            ],
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "seeds": list(checkpoint["seeds"]),
            "selected_final_epochs": int(checkpoint["selected_epochs"]),
            "pca_dims": checkpoint["pca_dims"],
            "support_alpha": checkpoint["support_alpha"],
            "leak_cap": checkpoint["leak_cap"],
            "delay_cap": checkpoint["delay_cap"],
            "power_alpha": checkpoint["power_alpha"],
            "projection_order": ["leaked PAS multi-marginal", "delay/PDP multi-marginal", "validation-gated tiny power"],
            "phase_policy": "preserve v209 local phase texture; no absolute phase regression or phase teacher",
        },
        "action_counts": {
            "leak_active": int(np.sum(actions["leak_alpha"].astype(np.float32) > 1e-7)),
            "delay_active": int(np.sum(actions["delay_alpha"].astype(np.float32) > 1e-7)),
            "power_active": int(np.sum(np.abs(actions["extra_log_delta"].astype(np.float32)) > 1e-8)),
            "by_regime": {
                regime: {
                    "count": int(np.sum(regimes == regime)),
                    "leak_active": int(np.sum((regimes == regime) & (actions["leak_alpha"].astype(np.float32) > 1e-7))),
                    "delay_active": int(np.sum((regimes == regime) & (actions["delay_alpha"].astype(np.float32) > 1e-7))),
                    "power_active": int(
                        np.sum((regimes == regime) & (np.abs(actions["extra_log_delta"].astype(np.float32)) > 1e-8))
                    ),
                }
                for regime in ("clear", "light", "medium", "heavy")
            },
        },
        "diagnostics": {
            "overall": {key: summarize(value) for key, value in action_diag.items()},
            "by_regime": by_regime_summary(
                regimes,
                {
                    "leak_alpha": actions["leak_alpha"].astype(np.float32),
                    "delay_alpha": actions["delay_alpha"].astype(np.float32),
                    "leaked_full_l1_actual": actions["leaked_full_l1_actual"].astype(np.float32),
                    "leaked_sh_l1_actual": actions["leaked_sh_l1_actual"].astype(np.float32),
                    "delay_full_l1_actual": actions["delay_full_l1_actual"].astype(np.float32),
                    "delay_d_l1_actual": actions["delay_d_l1_actual"].astype(np.float32),
                    "extra_log_delta": actions["extra_log_delta"].astype(np.float32),
                    "final_power_factor": actions["final_power_factor"].astype(np.float32),
                },
            ),
        },
        "comparison_to_v209": comparison_to_v209,
        "comparison_to_v215": comparison_to_v215,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "intermediate": PATHS.relative(INTERMEDIATE),
            "targets_cache": PATHS.relative(TARGETS_CACHE),
            "actions": PATHS.relative(ACTION_OUT),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(CHECKPOINT, CHECKPOINT.name)
        for src in [
            PATHS.code / "generate_v218_official_multispectrum_physics_ai.py",
            PATHS.code / "generate_v215_pointcloud_physics_neural_field.py",
            PATHS.code / "generate_v209_adaptive_metric_projection_controller.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
            PATHS.code / "generate_v187_leaked_pas_lifecycle_projection.py",
            PATHS.code / "build_official_pointcloud_scene.py",
            PATHS.code / "build_multibounce_path_candidates.py",
            PATHS.code / "analyze_geodesic_support_source.py",
            PATHS.code / "generate_v178_test_track_complex_consensus.py",
            PATHS.code / "generate_v87_easy_field_projection.py",
            PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py",
            PATHS.code / "generate_v169_power_scaled_tail_denoising.py",
            PATHS.code / "generate_v159_frontier_consensus_support_soup.py",
            PATHS.code / "powerfield_common.py",
            PATHS.code_root / "project_paths.py",
        ]:
            archive.write(src, src.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 20):
        block = np.asarray(output[start : start + 20], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    if OUTPUT.is_file() and ACTION_OUT.is_file() and CHECKPOINT.is_file():
        report = finalize_existing_report()
    else:
        report = generate()
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
