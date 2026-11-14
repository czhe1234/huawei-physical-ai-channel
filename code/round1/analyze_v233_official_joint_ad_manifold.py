#!/usr/bin/env python3
"""v233 analysis: official joint angle-delay residual manifold.

This probe learns the missing joint angle-delay support directly from released
training channels.  A point-cloud/path Transformer predicts PCA coefficients
of the log-support residual relative to an official train-H neighborhood
anchor.  The local v208 scorer is used only after generation as a diagnostic
gate; no local replay truth or Sionna-generated channel enters training or
generation.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import generate_v121_support_template_manifold as v121
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MP,
    MV,
    N_ANT,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
    summarize,
)
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 233
VERSION = "v233_official_joint_ad_manifold"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v233_official_joint_ad_manifold.json"
ACTION_OUT = OUT_DIR / "v233_official_joint_ad_actions.npz"
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BASE = PATHS.cache_dir(226) / "v226_early_shift_v224_path_heavy_Round1_Test_Channel.npy"
TRAIN_SUPPORT = PATHS.shared_features / "truth_coll_power.npy"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

NORMALIZED_SUPPORT = OUT_DIR / "v233_train_joint_support.npy"
TRAIN_ANCHOR = OUT_DIR / "v233_split_train_anchor.npy"
VAL_ANCHOR = OUT_DIR / "v233_split_val_anchor.npy"
FINAL_ANCHOR = OUT_DIR / "v233_final_train_anchor.npy"
TEST_ANCHOR = OUT_DIR / "v233_test_anchor.npy"
TEST_MODEL_TARGET = OUT_DIR / "v233_test_model_target.npy"

FLOOR = 1e-10
GATE_SCORE = 0.645604
PCA_DIMS = 112
VAL_COUNT = 320
ANCHOR_K = 72
BATCH_SIZE = 72
TRAIN_EPOCHS = 180
FINAL_SEEDS = (23301, 23317, 23337)
SUPPORT_BATCH = 8
PROJECT_BATCH = 10
REGIMES = ("clear", "light", "medium", "heavy")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_support(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid, None, None, None]
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a.reshape(len(a), -1).astype(np.float64)
    y = b.reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def load_normalized_train_support() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.load(TRAIN_SUPPORT, mmap_mode="r")
    raw_total = np.sum(np.asarray(raw), axis=(1, 2, 3), dtype=np.float64)
    valid = raw_total > 1e-20
    if not NORMALIZED_SUPPORT.is_file():
        out = np.lib.format.open_memmap(
            NORMALIZED_SUPPORT,
            mode="w+",
            dtype=np.float32,
            shape=raw.shape,
        )
        for start in range(0, len(raw), 32):
            end = min(start + 32, len(raw))
            out[start:end] = normalize_support(np.asarray(raw[start:end], dtype=np.float32))
            out.flush()
            print(f"v233 normalize train support {end}/{len(raw)}", flush=True)
    return np.load(NORMALIZED_SUPPORT, mmap_mode="r"), valid, np.log(np.maximum(raw_total, 1e-30)).astype(np.float32)


def weighted_support_average(
    bank: np.ndarray,
    idx: np.ndarray,
    weights: np.ndarray,
    output_path: Path,
) -> np.ndarray:
    if output_path.is_file():
        return np.load(output_path, mmap_mode="r")
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(idx), MH, MV, S_SC),
    )
    for start in range(0, len(idx), SUPPORT_BATCH):
        end = min(start + SUPPORT_BATCH, len(idx))
        block = np.zeros((end - start, MH, MV, S_SC), dtype=np.float32)
        for rank in range(idx.shape[1]):
            block += weights[start:end, rank, None, None, None] * np.asarray(
                bank[idx[start:end, rank]], dtype=np.float32
            )
        out[start:end] = normalize_support(block)
        out.flush()
        print(f"v233 anchor {output_path.name} {end}/{len(idx)}", flush=True)
    return np.load(output_path, mmap_mode="r")


def support_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred_beam = pred.sum(axis=3)
    truth_beam = truth.sum(axis=3)
    pred_delay = pred.sum(axis=(1, 2))
    truth_delay = truth.sum(axis=(1, 2))
    return {
        "joint": float(np.mean(row_cos(pred, truth))),
        "beam": float(np.mean(row_cos(pred_beam, truth_beam))),
        "delay": float(np.mean(row_cos(pred_delay, truth_delay))),
        "weighted": float(
            0.40 * np.mean(row_cos(pred, truth))
            + 0.28 * np.mean(row_cos(pred_beam, truth_beam))
            + 0.32 * np.mean(row_cos(pred_delay, truth_delay))
        ),
    }


def make_anchor_bundle(
    support: np.ndarray,
    valid_ids: np.ndarray,
    log_power: np.ndarray,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
) -> dict[str, object]:
    matched = choose_val_indices(train_pos[:, :2], test_pos[:, :2], VAL_COUNT, "matched-test", VERSION_NUMBER)
    val_ids = np.intersect1d(matched, valid_ids)
    train_ids = np.setdiff1d(valid_ids, val_ids)

    idx_train, w_train = v218.neighbor_index_weights(
        train_pos[train_ids],
        train_blocked[train_ids],
        train_pos[train_ids],
        train_blocked[train_ids],
        query_ids=train_ids,
        bank_ids=train_ids,
        k=ANCHOR_K,
    )
    idx_val, w_val = v218.neighbor_index_weights(
        train_pos[val_ids],
        train_blocked[val_ids],
        train_pos[train_ids],
        train_blocked[train_ids],
        k=ANCHOR_K,
    )
    split_bank = support[train_ids]
    train_anchor = weighted_support_average(split_bank, idx_train, w_train, TRAIN_ANCHOR)
    val_anchor = weighted_support_average(split_bank, idx_val, w_val, VAL_ANCHOR)
    train_anchor_power = np.sum(log_power[train_ids][idx_train] * w_train, axis=1).astype(np.float32)
    val_anchor_power = np.sum(log_power[train_ids][idx_val] * w_val, axis=1).astype(np.float32)

    idx_final, w_final = v218.neighbor_index_weights(
        train_pos[valid_ids],
        train_blocked[valid_ids],
        train_pos[valid_ids],
        train_blocked[valid_ids],
        query_ids=valid_ids,
        bank_ids=valid_ids,
        k=ANCHOR_K,
    )
    idx_test, w_test = v218.neighbor_index_weights(
        test_pos,
        test_blocked,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        k=ANCHOR_K,
    )
    final_bank = support[valid_ids]
    final_anchor = weighted_support_average(final_bank, idx_final, w_final, FINAL_ANCHOR)
    test_anchor = weighted_support_average(final_bank, idx_test, w_test, TEST_ANCHOR)
    final_anchor_power = np.sum(log_power[valid_ids][idx_final] * w_final, axis=1).astype(np.float32)
    test_anchor_power = np.sum(log_power[valid_ids][idx_test] * w_test, axis=1).astype(np.float32)
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "valid_ids": valid_ids,
        "train_anchor": train_anchor,
        "val_anchor": val_anchor,
        "final_anchor": final_anchor,
        "test_anchor": test_anchor,
        "train_anchor_power": train_anchor_power,
        "val_anchor_power": val_anchor_power,
        "final_anchor_power": final_anchor_power,
        "test_anchor_power": test_anchor_power,
        "w_train": w_train,
        "w_val": w_val,
        "w_final": w_final,
        "w_test": w_test,
    }


@dataclass
class FeatureSet:
    global_features: np.ndarray
    path_numeric: np.ndarray
    path_type: np.ndarray
    path_valid: np.ndarray


def raw_features(
    prefix: str,
    indices: np.ndarray | None,
    pos: np.ndarray,
    blocked: np.ndarray,
    anchor: np.ndarray,
    anchor_log_power: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beam = np.asarray(anchor).sum(axis=3).astype(np.float32)
    delay = np.asarray(anchor).sum(axis=(1, 2)).astype(np.float32)
    effective_count = (1.0 / np.maximum(np.sum(weights**2, axis=1), TINY)).astype(np.float32)
    agreement = np.max(weights, axis=1).astype(np.float32)
    anchors = v215.Anchors(
        beam=beam.reshape(len(anchor), -1),
        d=delay,
        du=np.zeros((len(anchor), N_ANT, S_SC), dtype=np.float32),
        dh=np.zeros((len(anchor), MH, S_SC), dtype=np.float32),
        log_power=anchor_log_power,
        agreement=agreement,
        effective_count=effective_count,
    )
    global_base = v215.make_global_features(prefix, indices, pos, blocked, anchors)
    signature = v121.support_signature(np.asarray(anchor, dtype=np.float32), FLOOR)
    global_raw = np.concatenate([global_base, signature], axis=1).astype(np.float32)
    path_numeric, path_type, path_valid = v215.make_path_tokens(prefix, pos, indices)
    return global_raw, path_numeric, path_type, path_valid


def standardize_feature_sets(
    train_raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    others: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[FeatureSet, list[FeatureSet], dict[str, np.ndarray]]:
    train_global, train_path, train_type, train_valid = train_raw
    global_mean = train_global.mean(axis=0).astype(np.float32)
    global_std = np.maximum(train_global.std(axis=0), 1e-5).astype(np.float32)
    path_values = train_path[train_valid]
    path_mean = path_values.mean(axis=0).astype(np.float32)
    path_std = np.maximum(path_values.std(axis=0), 1e-5).astype(np.float32)

    def convert(raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> FeatureSet:
        glob, path, ptype, valid = raw
        glob_n = np.clip((glob - global_mean) / global_std, -8.0, 8.0).astype(np.float32)
        path_n = np.clip((path - path_mean[None, None, :]) / path_std[None, None, :], -8.0, 8.0).astype(np.float32)
        path_n[~valid] = 0.0
        return FeatureSet(glob_n, path_n, ptype.astype(np.int64), valid.astype(bool))

    stats = {
        "global_mean": global_mean,
        "global_std": global_std,
        "path_mean": path_mean,
        "path_std": path_std,
    }
    return convert(train_raw), [convert(value) for value in others], stats


class JointSupportField(nn.Module):
    def __init__(self, global_dim: int, path_dim: int, coeff_dim: int):
        super().__init__()
        d_model = 192
        self.type_emb = nn.Embedding(4, 24)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, 448),
            nn.SiLU(),
            nn.LayerNorm(448),
            nn.Dropout(0.06),
            nn.Linear(448, d_model),
        )
        self.path_proj = nn.Sequential(
            nn.Linear(path_dim + 24, 288),
            nn.SiLU(),
            nn.LayerNorm(288),
            nn.Linear(288, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=6,
            dim_feedforward=768,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=5)
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 448),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(448, 320),
            nn.SiLU(),
            nn.LayerNorm(320),
        )
        self.head = nn.Linear(320, coeff_dim)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
    ) -> torch.Tensor:
        query = self.global_proj(global_features).unsqueeze(1)
        path = self.path_proj(torch.cat([path_numeric, self.type_emb(path_type)], dim=2))
        tokens = torch.cat([query, path], dim=1)
        mask = torch.cat(
            [torch.zeros((len(path_valid), 1), dtype=torch.bool, device=path_valid.device), ~path_valid],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        return self.head(self.trunk(encoded[:, 0]))


def make_coeff(anchor: np.ndarray, target: np.ndarray, basis: v215.Basis) -> np.ndarray:
    residual = np.clip(
        np.log(np.maximum(target.reshape(len(target), -1), FLOOR))
        - np.log(np.maximum(anchor.reshape(len(anchor), -1), FLOOR)),
        -8.0,
        8.0,
    ).astype(np.float32)
    raw = (residual - basis.mean[None, :]) @ basis.components.T
    return (raw / basis.coeff_scale[None, :]).astype(np.float32)


def reconstruct_torch(
    coeff: torch.Tensor,
    anchor: torch.Tensor,
    basis_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    mean, components, scale = basis_tensors
    delta = (coeff * scale[None, :]) @ components + mean[None, :]
    logits = torch.log(anchor.reshape(len(anchor), -1).clamp_min(FLOOR)) + delta
    return torch.softmax(logits, dim=1)


def weighted_loss(pred: torch.Tensor, truth: torch.Tensor, coeff: torch.Tensor, coeff_truth: torch.Tensor) -> torch.Tensor:
    p = pred.reshape(-1, MH, MV, S_SC)
    t = truth.reshape(-1, MH, MV, S_SC)
    joint_loss = 1.0 - F.cosine_similarity(pred, truth, dim=1).mean()
    beam_loss = 1.0 - F.cosine_similarity(p.sum(dim=3).reshape(len(p), -1), t.sum(dim=3).reshape(len(t), -1), dim=1).mean()
    delay_loss = 1.0 - F.cosine_similarity(p.sum(dim=(1, 2)), t.sum(dim=(1, 2)), dim=1).mean()
    hellinger = 1.0 - torch.sum(torch.sqrt(pred.clamp_min(FLOOR) * truth.clamp_min(FLOOR)), dim=1).mean()
    coeff_loss = F.smooth_l1_loss(coeff, coeff_truth, beta=0.55)
    return 0.36 * joint_loss + 0.25 * beam_loss + 0.27 * delay_loss + 0.08 * hellinger + 0.04 * coeff_loss


@torch.no_grad()
def predict_model(
    model: JointSupportField,
    features: FeatureSet,
    anchor: np.ndarray,
    basis: v215.Basis,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    basis_tensors = (
        torch.from_numpy(basis.mean).to(device),
        torch.from_numpy(basis.components).to(device),
        torch.from_numpy(basis.coeff_scale).to(device),
    )
    out = np.empty((len(anchor), MH, MV, S_SC), dtype=np.float32)
    for start in range(0, len(anchor), 96):
        end = min(start + 96, len(anchor))
        coeff = model(
            torch.from_numpy(features.global_features[start:end]).to(device),
            torch.from_numpy(features.path_numeric[start:end]).to(device),
            torch.from_numpy(features.path_type[start:end]).to(device),
            torch.from_numpy(features.path_valid[start:end]).to(device),
        )
        pred = reconstruct_torch(
            coeff,
            torch.from_numpy(np.asarray(anchor[start:end], dtype=np.float32)).to(device),
            basis_tensors,
        )
        out[start:end] = pred.cpu().numpy().reshape(end - start, MH, MV, S_SC)
    return out


def train_model(
    features: FeatureSet,
    anchor: np.ndarray,
    target: np.ndarray,
    coeff_truth: np.ndarray,
    basis: v215.Basis,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[FeatureSet, np.ndarray, np.ndarray] | None,
) -> tuple[JointSupportField, dict[str, object]]:
    set_seed(seed)
    model = JointSupportField(features.global_features.shape[1], features.path_numeric.shape[2], len(basis.coeff_scale)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.8e-4, weight_decay=2.8e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    basis_tensors = (
        torch.from_numpy(basis.mean).to(device),
        torch.from_numpy(basis.components).to(device),
        torch.from_numpy(basis.coeff_scale).to(device),
    )
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(len(anchor))
        for start in range(0, len(anchor), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            coeff = model(
                torch.from_numpy(features.global_features[ids]).to(device),
                torch.from_numpy(features.path_numeric[ids]).to(device),
                torch.from_numpy(features.path_type[ids]).to(device),
                torch.from_numpy(features.path_valid[ids]).to(device),
            )
            anchor_t = torch.from_numpy(np.asarray(anchor[ids], dtype=np.float32)).to(device)
            truth_t = torch.from_numpy(np.asarray(target[ids], dtype=np.float32).reshape(len(ids), -1)).to(device)
            pred = reconstruct_torch(coeff, anchor_t, basis_tensors)
            coeff_t = torch.from_numpy(coeff_truth[ids]).to(device)
            loss = weighted_loss(pred, truth_t, coeff, coeff_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if validation is not None:
                val_features, val_anchor, val_truth = validation
                val_pred = predict_model(model, val_features, val_anchor, basis, device)
                metrics = support_metrics(val_pred, val_truth)
                anchor_metrics = support_metrics(np.asarray(val_anchor), val_truth)
                item["val_model"] = metrics
                item["val_anchor"] = anchor_metrics
                if metrics["weighted"] > best_metric:
                    best_metric = metrics["weighted"]
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v233 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": None if not np.isfinite(best_metric) else float(best_metric),
        "history": history,
    }


def calibrate_alpha(
    anchor: np.ndarray,
    pred: np.ndarray,
    truth: np.ndarray,
    regimes: np.ndarray,
) -> tuple[dict[str, float], dict[str, object]]:
    grid = np.linspace(0.0, 1.25, 26, dtype=np.float32)
    selected: dict[str, float] = {}
    report: dict[str, object] = {}
    for regime in REGIMES:
        mask = regimes.astype(str) == regime
        if not np.any(mask):
            selected[regime] = 0.0
            continue
        scores = []
        for alpha in grid:
            blend = normalize_support((1.0 - alpha) * np.asarray(anchor[mask]) + alpha * pred[mask])
            scores.append(support_metrics(blend, truth[mask])["weighted"])
        best = int(np.argmax(scores))
        selected[regime] = float(grid[best])
        report[regime] = {
            "count": int(np.sum(mask)),
            "alpha": float(grid[best]),
            "anchor_score": float(scores[0]),
            "best_score": float(scores[best]),
        }
    return selected, report


def extract_base_support(pos_test: np.ndarray) -> np.ndarray:
    path = OUT_DIR / "v233_v226_current_support.npy"
    if path.is_file():
        return np.load(path, mmap_mode="r")
    source = np.load(BASE, mmap_mode="r")
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(source), MH, MV, S_SC))
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1).astype(np.float32)
    beta_s = beta_subcarriers(140.3, 0.10)
    for start in range(0, len(source), PROJECT_BATCH):
        end = min(start + PROJECT_BATCH, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        derot = h * np.exp(1j * distance[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad = ad_transform(derot)
        coll = (np.abs(ad) ** 2).astype(np.float32).sum(axis=(1, 4))
        out[start:end] = normalize_support(coll)
        out.flush()
        print(f"v233 extract v226 support {end}/{len(source)}", flush=True)
    return np.load(path, mmap_mode="r")


def apply_target(
    name: str,
    target: np.ndarray,
    alpha: np.ndarray,
    pos_test: np.ndarray,
    mode: str,
    output_dir: Path | None = None,
    version_prefix: str = "v233",
) -> tuple[Path, dict[str, object]]:
    source = np.load(BASE, mmap_mode="r")
    output = (OUT_DIR if output_dir is None else output_dir) / f"{version_prefix}_{name}_Round1_Test_Channel.npy"
    out = np.lib.format.open_memmap(output, mode="w+", dtype=np.complex64, shape=source.shape)
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1).astype(np.float32)
    beta_s = beta_subcarriers(140.3, 0.10)
    diag = {"alpha": alpha.astype(np.float32), "support_l1": [], "target_cos": [], "actual_l1": []}
    for start in range(0, len(source), PROJECT_BATCH):
        end = min(start + PROJECT_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * distance[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        power0 = (np.abs(ad0) ** 2).astype(np.float32)
        coll0 = power0.sum(axis=(1, 4))
        total0 = np.maximum(coll0.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        current = coll0 / total0[:, None, None, None]
        tgt = np.asarray(target[start:end], dtype=np.float32)
        a = alpha[start:end, None, None, None]
        if mode == "log":
            desired = np.exp((1.0 - a) * np.log(np.maximum(current, FLOOR)) + a * np.log(np.maximum(tgt, FLOOR))).astype(np.float32)
            desired = normalize_support(desired)
        else:
            desired = normalize_support((1.0 - a) * current + a * tgt)
        gain = np.sqrt(desired / np.maximum(current, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.26, 4.20)
        flat = current.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(end - start, MH, MV, S_SC)
        strong1024 = top_mask(flat, 1024).reshape(end - start, MH, MV, S_SC)
        strong8192 = top_mask(flat, 8192).reshape(end - start, MH, MV, S_SC)
        gain[strong8192] = np.clip(gain[strong8192], 0.54, 2.15)
        gain[strong1024] = np.clip(gain[strong1024], 0.70, 1.58)
        gain[strong128] = np.clip(gain[strong128], 0.84, 1.24)
        ad1 = ad0 * gain[:, None, :, :, None, :]
        total1 = np.maximum(np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64), TINY)
        ad1 *= np.sqrt(total0 / total1)[:, None, None, None, None, None].astype(np.float32)
        h1 = inverse_ad(ad1).reshape(end - start, MP * MH * MV, N_ANT, S_SC)
        h1 *= np.exp(-1j * distance[start:end, None, None, None] * beta_s).astype(np.complex64)
        out[start:end] = h1.astype(np.complex64)
        out.flush()
        coll1 = (np.abs(ad1) ** 2).astype(np.float32).sum(axis=(1, 4))
        support1 = normalize_support(coll1)
        diag["support_l1"].append(np.sum(np.abs(desired - current), axis=(1, 2, 3)).astype(np.float32))
        diag["target_cos"].append(row_cos(current, tgt))
        diag["actual_l1"].append(np.sum(np.abs(support1 - current), axis=(1, 2, 3)).astype(np.float32))
        print(f"{version_prefix} apply {name} {end}/{len(source)}", flush=True)
    return output, {
        "alpha": summarize(alpha),
        "support_l1": summarize(np.concatenate(diag["support_l1"])),
        "target_cos": summarize(np.concatenate(diag["target_cos"])),
        "actual_l1": summarize(np.concatenate(diag["actual_l1"])),
    }


def main() -> None:
    for path in [BASE, TRAIN_SUPPORT, GEOM_CACHE, v215.SCENE_CACHE, v215.GEODESIC_CACHE, v215.PATH_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v233 device={device}", flush=True)
    support, valid, log_power = load_normalized_train_support()
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    bundle = make_anchor_bundle(support, valid_ids, log_power, train_pos, test_pos, train_blocked, test_blocked)
    train_ids = bundle["train_ids"]
    val_ids = bundle["val_ids"]

    train_target = np.asarray(support[train_ids], dtype=np.float32)
    val_target = np.asarray(support[val_ids], dtype=np.float32)
    split_basis, split_coeff = v215.fit_basis(np.asarray(bundle["train_anchor"]), train_target, PCA_DIMS)
    train_raw = raw_features(
        "train", train_ids, train_pos[train_ids], train_blocked[train_ids], bundle["train_anchor"], bundle["train_anchor_power"], bundle["w_train"]
    )
    val_raw = raw_features(
        "train", val_ids, train_pos[val_ids], train_blocked[val_ids], bundle["val_anchor"], bundle["val_anchor_power"], bundle["w_val"]
    )
    train_features, [val_features], split_stats = standardize_feature_sets(train_raw, [val_raw])
    validation_model, validation_report = train_model(
        train_features,
        bundle["train_anchor"],
        train_target,
        split_coeff,
        split_basis,
        FINAL_SEEDS[0],
        TRAIN_EPOCHS,
        device,
        (val_features, bundle["val_anchor"], val_target),
    )
    val_prediction = predict_model(validation_model, val_features, bundle["val_anchor"], split_basis, device)
    val_regimes = regime_of(train_blocked[val_ids])
    val_alpha, calibration_report = calibrate_alpha(bundle["val_anchor"], val_prediction, val_target, val_regimes)
    validation_metrics = {
        "anchor": support_metrics(np.asarray(bundle["val_anchor"]), val_target),
        "model": support_metrics(val_prediction, val_target),
    }
    best_epoch = max(40, int(validation_report["best_epoch"]))
    del validation_model, val_prediction, split_basis, split_coeff
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    final_target = np.asarray(support[valid_ids], dtype=np.float32)
    final_basis, final_coeff = v215.fit_basis(np.asarray(bundle["final_anchor"]), final_target, PCA_DIMS)
    final_raw = raw_features(
        "train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids], bundle["final_anchor"], bundle["final_anchor_power"], bundle["w_final"]
    )
    test_raw = raw_features(
        "test", None, test_pos, test_blocked, bundle["test_anchor"], bundle["test_anchor_power"], bundle["w_test"]
    )
    final_features, [test_features], final_stats = standardize_feature_sets(final_raw, [test_raw])
    seed_predictions = []
    final_training = []
    for seed in FINAL_SEEDS:
        model, info = train_model(
            final_features,
            bundle["final_anchor"],
            final_target,
            final_coeff,
            final_basis,
            seed,
            best_epoch,
            device,
            None,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "basis": final_basis,
                "feature_stats": final_stats,
                "seed": seed,
                "epochs": best_epoch,
            },
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        seed_predictions.append(predict_model(model, test_features, bundle["test_anchor"], final_basis, device))
        final_training.append(info)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log_mean = np.mean(np.stack([np.log(np.maximum(value, FLOOR)) for value in seed_predictions]), axis=0)
    test_model_target = normalize_support(np.exp(log_mean).astype(np.float32))
    np.save(TEST_MODEL_TARGET, test_model_target)
    seed_pair_cos = []
    for i in range(len(seed_predictions)):
        for j in range(i + 1, len(seed_predictions)):
            seed_pair_cos.append(row_cos(seed_predictions[i], seed_predictions[j]))
    ensemble_agreement = np.mean(np.stack(seed_pair_cos), axis=0).astype(np.float32)

    current_support = extract_base_support(test_pos)
    regimes = regime_of(test_blocked)
    alpha_base = np.asarray([val_alpha[str(regime)] for regime in regimes], dtype=np.float32)
    confidence = np.clip((ensemble_agreement - 0.84) / 0.13, 0.0, 1.0).astype(np.float32)
    confidence *= np.clip((row_cos(test_model_target, np.asarray(bundle["test_anchor"])) - 0.72) / 0.22, 0.0, 1.0).astype(np.float32)
    variants = [
        ("linear_calibrated", np.clip(alpha_base * (0.40 + 0.60 * confidence), 0.0, 0.95), "linear"),
        ("log_calibrated", np.clip(alpha_base * (0.34 + 0.66 * confidence), 0.0, 0.82), "log"),
        ("log_aggressive", np.clip(1.35 * alpha_base * (0.42 + 0.58 * confidence), 0.0, 1.08), "log"),
        (
            "heavy_joint_overdrive",
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.08, "medium": 0.48, "heavy": 0.74}[str(r)] for r in regimes], dtype=np.float32)
                * (0.48 + 0.52 * confidence),
                0.0,
                0.90,
            ),
            "log",
        ),
    ]
    results = []
    for name, alpha, mode in variants:
        output, diagnostics = apply_target(name, test_model_target, alpha.astype(np.float32), test_pos, mode)
        results.append(
            {
                "name": name,
                "mode": mode,
                "output": PATHS.relative(output),
                "score": v208.score_prediction(output),
                "diagnostics": diagnostics,
            }
        )
    best = max(results, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        ensemble_agreement=ensemble_agreement,
        confidence=confidence,
        model_vs_test_anchor=row_cos(test_model_target, np.asarray(bundle["test_anchor"])),
        model_vs_v226=row_cos(test_model_target, np.asarray(current_support)),
        alpha_base=alpha_base,
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Official train H contains a learnable joint angle-delay residual manifold that marginal-only models and test-output self-tracking miss.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H",
                "official train/test coordinates",
                "official point-cloud scene/geodesic features",
                "official point-cloud multibounce path candidates",
                "v226 strict official-data carrier",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217/v221/v222 teacher outputs",
                "synthetic replay truth",
                "hidden online labels",
                "raw RT H",
            ],
            "note": "The local replay scorer is evaluation-only. No ZIP is created below the v212 local gate.",
        },
        "training": {
            "device": str(device),
            "valid_train_count": int(len(valid_ids)),
            "split_train_count": int(len(train_ids)),
            "validation_count": int(len(val_ids)),
            "pca_dims": PCA_DIMS,
            "best_epoch": best_epoch,
            "validation": validation_report,
            "validation_metrics": validation_metrics,
            "calibration": calibration_report,
            "alpha_by_regime": val_alpha,
            "final_seeds": list(FINAL_SEEDS),
            "final_training": final_training,
            "split_feature_dim": int(train_features.global_features.shape[1]),
            "path_dim": int(train_features.path_numeric.shape[2]),
        },
        "test_diagnostics": {
            "ensemble_agreement": summarize(ensemble_agreement),
            "confidence": summarize(confidence),
            "model_vs_anchor": summarize(row_cos(test_model_target, np.asarray(bundle["test_anchor"]))),
            "model_vs_v226": summarize(row_cos(test_model_target, np.asarray(current_support))),
            "anchor_vs_v226": summarize(row_cos(np.asarray(bundle["test_anchor"]), np.asarray(current_support))),
        },
        "variants": results,
        "local_gate": {
            "required_score_traceback_leaked": GATE_SCORE,
            "best_variant": best["name"],
            "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
            "gap_to_gate": best["score"]["score_traceback_leaked"] - GATE_SCORE,
        },
        "outputs": {
            "model_target": PATHS.relative(TEST_MODEL_TARGET),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
