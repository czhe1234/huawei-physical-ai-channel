#!/usr/bin/env python3
"""v252: official-H-supervised calibration of point-cloud Sionna path fields."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 252
VERSION = "v252_pointcloud_pathfield_calibrator"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v252_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v252_test_calibrated_pdp.npy"
ACTION_OUT = OUT_DIR / "v252_pathfield_actions.npz"

FIELD_CACHE = OUT_DIR / "v252_pointcloud_sionna_path_fields.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
GEODESIC_CACHE = PATHS.shared_features / "geodesic_support_features.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"

S_SC = 192
VAL_COUNT = 320
ANCHOR_K = 24
EPOCHS = 60
SEEDS = (25201, 25231, 25261)
LOCAL_GATE = 0.645604
FLOOR = 1e-11


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    out /= np.maximum(out.sum(axis=1, keepdims=True), TINY)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def make_anchor(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
    bank_d: np.ndarray,
    *,
    query_ids: np.ndarray | None = None,
    bank_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    idx, weights = v218.neighbor_index_weights(
        query_pos,
        query_blocked,
        bank_pos,
        bank_blocked,
        query_ids=query_ids,
        bank_ids=bank_ids,
        k=ANCHOR_K,
    )
    return normalize(v218.weighted_average(bank_d, idx, weights)), weights.astype(np.float32)


@dataclass
class Features:
    global_features: np.ndarray
    path_numeric: np.ndarray
    path_type: np.ndarray
    path_valid: np.ndarray
    raw_delay: np.ndarray
    log_field_power: np.ndarray


def field_features(
    cache: np.lib.npyio.NpzFile,
    prefix: str,
    indices: np.ndarray | None,
    pos: np.ndarray,
    blocked: np.ndarray,
    anchor: np.ndarray,
    weights: np.ndarray,
) -> Features:
    def take(name: str) -> np.ndarray:
        value = cache[f"{prefix}_{name}"]
        return value if indices is None else value[indices]

    valid = take("valid").astype(bool)
    delay = np.nan_to_num(take("delay_bin_physical").astype(np.float32), nan=0.0)
    length = np.nan_to_num(take("path_length").astype(np.float32), nan=0.0, posinf=600.0)
    theta_t = np.nan_to_num(take("theta_t").astype(np.float32), nan=0.0)
    phi_t = np.nan_to_num(take("phi_t").astype(np.float32), nan=0.0)
    theta_r = np.nan_to_num(take("theta_r").astype(np.float32), nan=0.0)
    phi_r = np.nan_to_num(take("phi_r").astype(np.float32), nan=0.0)
    count = take("interaction_count").astype(np.float32)
    mask = take("interaction_mask").astype(np.int16)
    score = np.nan_to_num(take("score").astype(np.float32), nan=-8.0, neginf=-8.0, posinf=8.0)
    power = np.nan_to_num(take("path_power").astype(np.float32), nan=0.0)
    log_power = np.log(np.maximum(power, 1e-40)).astype(np.float32)
    max_log = np.max(np.where(valid, log_power, -100.0), axis=1, keepdims=True)
    relative_log_power = np.where(valid, np.clip(log_power - max_log, -24.0, 0.0), 0.0)
    bits = np.stack([((mask & bit) != 0).astype(np.float32) for bit in (1, 2, 4, 8)], axis=2)
    numeric = np.concatenate(
        [
            valid[..., None].astype(np.float32),
            (delay / 24.0)[..., None],
            (np.log1p(length) / 7.0)[..., None],
            (relative_log_power / 12.0)[..., None],
            (score / 5.0)[..., None],
            (count / 4.0)[..., None],
            bits,
            np.stack(
                [
                    np.sin(theta_t), np.cos(theta_t), np.sin(phi_t), np.cos(phi_t),
                    np.sin(theta_r), np.cos(theta_r), np.sin(phi_r), np.cos(phi_r),
                ],
                axis=2,
            ),
        ],
        axis=2,
    ).astype(np.float32)
    type_id = np.zeros_like(mask, dtype=np.int64)
    type_id[valid & (count == 0)] = 1
    type_id[valid & ((mask & 8) != 0)] = 5
    type_id[valid & ((mask & 4) != 0) & (type_id == 0)] = 4
    type_id[valid & ((mask & 2) != 0) & (type_id == 0)] = 3
    type_id[valid & ((mask & 1) != 0) & (type_id == 0)] = 2
    type_id[valid & (type_id == 0)] = 6

    scene = np.load(SCENE_CACHE)[f"{prefix}_features"].astype(np.float32)
    geodesic = np.load(GEODESIC_CACHE)[f"{prefix}_features"].astype(np.float32)
    if indices is not None:
        scene = scene[indices]
        geodesic = geodesic[indices]
    regimes = regime_of(blocked).astype(str)
    onehot = np.stack(
        [(regimes == name).astype(np.float32) for name in ("clear", "light", "medium", "heavy")],
        axis=1,
    )
    effective = 1.0 / np.maximum(np.sum(weights**2, axis=1), 1e-8)
    global_features = np.concatenate(
        [
            v215.position_features(pos),
            (blocked / 160.0)[:, None].astype(np.float32),
            onehot,
            np.nan_to_num(scene, nan=0.0, posinf=8.0, neginf=-8.0),
            np.nan_to_num(geodesic, nan=0.0, posinf=8.0, neginf=-8.0),
            v215.d_stats(anchor),
            np.max(weights, axis=1)[:, None].astype(np.float32),
            (effective / ANCHOR_K)[:, None].astype(np.float32),
            valid.mean(axis=1, keepdims=True).astype(np.float32),
            np.log10(np.maximum(power.sum(axis=1), 1e-40))[:, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return Features(
        np.nan_to_num(global_features, nan=0.0, posinf=8.0, neginf=-8.0),
        np.nan_to_num(numeric, nan=0.0, posinf=8.0, neginf=-8.0),
        type_id,
        valid,
        delay,
        relative_log_power,
    )


def standardize(train: Features, others: list[Features]) -> tuple[Features, list[Features], dict[str, np.ndarray]]:
    g_mean = train.global_features.mean(axis=0).astype(np.float32)
    g_std = np.maximum(train.global_features.std(axis=0), 1e-5).astype(np.float32)
    p_values = train.path_numeric[train.path_valid]
    p_mean = p_values.mean(axis=0).astype(np.float32)
    p_std = np.maximum(p_values.std(axis=0), 1e-5).astype(np.float32)

    def convert(value: Features) -> Features:
        glob = np.clip((value.global_features - g_mean) / g_std, -8.0, 8.0).astype(np.float32)
        path = np.clip((value.path_numeric - p_mean[None, None]) / p_std[None, None], -8.0, 8.0).astype(np.float32)
        path[~value.path_valid] = 0.0
        return Features(glob, path, value.path_type, value.path_valid, value.raw_delay, value.log_field_power)

    return convert(train), [convert(value) for value in others], {
        "global_mean": g_mean,
        "global_std": g_std,
        "path_mean": p_mean,
        "path_std": p_std,
    }


class PathFieldCalibrator(nn.Module):
    def __init__(self, global_dim: int, path_dim: int):
        super().__init__()
        width = 160
        self.type_emb = nn.Embedding(7, 20)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, 384), nn.SiLU(), nn.LayerNorm(384), nn.Linear(384, width)
        )
        self.path_proj = nn.Sequential(
            nn.Linear(path_dim + 20, 256), nn.SiLU(), nn.LayerNorm(256), nn.Linear(256, width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=5,
            dim_feedforward=640,
            dropout=0.07,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.path_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 192), nn.SiLU())
        self.global_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 320), nn.SiLU())
        self.energy_residual = nn.Linear(192, 1)
        self.delay_offset = nn.Linear(192, 1)
        self.delay_width = nn.Linear(192, 1)
        self.anchor_residual = nn.Linear(320, S_SC)
        self.mix_gate = nn.Linear(320, 1)
        self.field_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        nn.init.zeros_(self.energy_residual.weight)
        nn.init.zeros_(self.energy_residual.bias)
        nn.init.zeros_(self.anchor_residual.weight)
        nn.init.zeros_(self.anchor_residual.bias)
        nn.init.constant_(self.mix_gate.bias, -2.0)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        raw_delay: torch.Tensor,
        log_field_power: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.global_proj(global_features).unsqueeze(1)
        path = self.path_proj(torch.cat([path_numeric, self.type_emb(path_type)], dim=2))
        tokens = torch.cat([query, path], dim=1)
        padding = torch.cat(
            [torch.zeros((len(path_valid), 1), dtype=torch.bool, device=path_valid.device), ~path_valid], dim=1
        )
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        global_latent = self.global_head(encoded[:, 0])
        path_latent = self.path_head(encoded[:, 1:])
        center = raw_delay + 2.5 * torch.tanh(self.delay_offset(path_latent).squeeze(2))
        width = 0.55 + 3.5 * torch.sigmoid(self.delay_width(path_latent).squeeze(2))
        bins = torch.arange(S_SC, dtype=torch.float32, device=anchor.device)[None, None]
        kernel = torch.exp(-0.5 * ((bins - center[..., None]) / width[..., None]) ** 2)
        kernel = kernel * path_valid[..., None]
        logits = (
            F.softplus(self.field_scale) * log_field_power
            + 1.8 * torch.tanh(self.energy_residual(path_latent).squeeze(2))
        ).masked_fill(~path_valid, -1e9)
        weights = torch.softmax(logits, dim=1) * path_valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-30)
        path_d = torch.sum(weights[..., None] * kernel, dim=1)
        path_d = path_d / path_d.sum(dim=1, keepdim=True).clamp_min(1e-30)
        no_path = ~torch.any(path_valid, dim=1)
        path_d = torch.where(no_path[:, None], anchor, path_d)

        residual = 1.6 * torch.tanh(self.anchor_residual(global_latent))
        anchor_d = torch.softmax(torch.log(anchor.clamp_min(FLOOR)) + residual, dim=1)
        gate = 0.82 * torch.sigmoid(self.mix_gate(global_latent).squeeze(1))
        gate = torch.where(no_path, torch.zeros_like(gate), gate)
        pred = (1.0 - gate[:, None]) * anchor_d + gate[:, None] * path_d
        pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return pred, path_d, gate


@torch.no_grad()
def predict(
    model: PathFieldCalibrator,
    features: Features,
    anchor: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    pred = np.empty_like(anchor, dtype=np.float32)
    path_d = np.empty_like(anchor, dtype=np.float32)
    gate = np.empty(len(anchor), dtype=np.float32)
    for start in range(0, len(anchor), 64):
        end = min(start + 64, len(anchor))
        out, phys, g = model(
            torch.from_numpy(features.global_features[start:end]).to(device),
            torch.from_numpy(features.path_numeric[start:end]).to(device),
            torch.from_numpy(features.path_type[start:end]).to(device),
            torch.from_numpy(features.path_valid[start:end]).to(device),
            torch.from_numpy(features.raw_delay[start:end]).to(device),
            torch.from_numpy(features.log_field_power[start:end]).to(device),
            torch.from_numpy(anchor[start:end]).to(device),
        )
        pred[start:end] = out.cpu().numpy()
        path_d[start:end] = phys.cpu().numpy()
        gate[start:end] = g.cpu().numpy()
    return normalize(pred), normalize(path_d), gate


def train_model(
    train_features: Features,
    train_anchor: np.ndarray,
    train_target: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[Features, np.ndarray, np.ndarray] | None,
) -> tuple[PathFieldCalibrator, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PathFieldCalibrator(train_features.global_features.shape[1], train_features.path_numeric.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.5e-4)
    best_state = None
    best_epoch = epochs
    best_cos = -np.inf
    history = []
    batch_size = 64
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_target))
        losses = []
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            anchor_t = torch.from_numpy(train_anchor[ids]).to(device)
            target_t = torch.from_numpy(train_target[ids]).to(device)
            out, _path, gate = model(
                torch.from_numpy(train_features.global_features[ids]).to(device),
                torch.from_numpy(train_features.path_numeric[ids]).to(device),
                torch.from_numpy(train_features.path_type[ids]).to(device),
                torch.from_numpy(train_features.path_valid[ids]).to(device),
                torch.from_numpy(train_features.raw_delay[ids]).to(device),
                torch.from_numpy(train_features.log_field_power[ids]).to(device),
                anchor_t,
            )
            cosine = F.cosine_similarity(out, target_t, dim=1)
            log_l1 = torch.mean(torch.abs(torch.log(out.clamp_min(FLOOR)) - torch.log(target_t.clamp_min(FLOOR))), dim=1)
            loss = torch.mean(1.0 - cosine + 0.015 * log_l1 + 0.006 * gate)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if validation is not None and (epoch == 1 or epoch % 10 == 0):
            val_features, val_anchor, val_target = validation
            val_pred, _val_path, val_gate = predict(model, val_features, val_anchor, device)
            val_cos = float(np.mean(row_cos(val_pred, val_target)))
            item = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_cos": val_cos,
                "val_gate_mean": float(np.mean(val_gate)),
            }
            history.append(item)
            print(f"v252 seed={seed} {json.dumps(item)}", flush=True)
            if val_cos > best_cos:
                best_cos = val_cos
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "best_cos": best_cos, "history": history}


def calibrate_alpha(
    anchor: np.ndarray,
    pred: np.ndarray,
    truth: np.ndarray,
    regimes: np.ndarray,
) -> tuple[dict[str, float], dict[str, object]]:
    alpha_by_regime = {}
    report = {}
    for regime in ("clear", "light", "medium", "heavy"):
        mask = regimes.astype(str) == regime
        base = float(np.mean(row_cos(anchor[mask], truth[mask])))
        best = (base, 0.0)
        curve = []
        for alpha in np.linspace(0.0, 1.0, 21):
            value = normalize((1.0 - alpha) * anchor[mask] + alpha * pred[mask])
            score = float(np.mean(row_cos(value, truth[mask])))
            curve.append({"alpha": float(alpha), "cos": score})
            if score > best[0]:
                best = (score, float(alpha))
        alpha_by_regime[regime] = best[1]
        report[regime] = {"count": int(np.sum(mask)), "anchor_cos": base, "best_cos": best[0], "alpha": best[1], "curve": curve}
    return alpha_by_regime, report


def main() -> None:
    for path in [FIELD_CACHE, TARGET_CACHE, GEOM_CACHE, SCENE_CACHE, GEODESIC_CACHE, BASE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fields = np.load(FIELD_CACHE)
    targets = np.load(TARGET_CACHE)
    d_all = normalize(targets["d"].astype(np.float32))
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    matched = choose_val_indices(train_pos[:, :2], test_pos[:, :2], VAL_COUNT, "matched-test", VERSION_NUMBER)
    val_ids = np.intersect1d(matched, valid_ids)
    train_ids = np.setdiff1d(valid_ids, val_ids)

    train_anchor, train_w = make_anchor(
        train_pos[train_ids], train_blocked[train_ids], train_pos[train_ids], train_blocked[train_ids], d_all[train_ids],
        query_ids=train_ids, bank_ids=train_ids,
    )
    val_anchor, val_w = make_anchor(
        train_pos[val_ids], train_blocked[val_ids], train_pos[train_ids], train_blocked[train_ids], d_all[train_ids]
    )
    train_raw = field_features(fields, "train", train_ids, train_pos[train_ids], train_blocked[train_ids], train_anchor, train_w)
    val_raw = field_features(fields, "train", val_ids, train_pos[val_ids], train_blocked[val_ids], val_anchor, val_w)
    train_features, [val_features], _split_stats = standardize(train_raw, [val_raw])
    model, validation = train_model(
        train_features, train_anchor, d_all[train_ids], SEEDS[0], EPOCHS, device,
        (val_features, val_anchor, d_all[val_ids]),
    )
    val_pred, val_path, val_gate = predict(model, val_features, val_anchor, device)
    val_regimes = regime_of(train_blocked[val_ids])
    alpha_by_regime, calibration = calibrate_alpha(val_anchor, val_pred, d_all[val_ids], val_regimes)
    val_alpha = np.asarray(
        [alpha_by_regime[str(regime)] for regime in val_regimes.astype(str)],
        dtype=np.float32,
    )
    calibrated_val = normalize(
        (1.0 - val_alpha[:, None]) * val_anchor + val_alpha[:, None] * val_pred
    )
    validation_metrics = {
        "anchor_cos": float(np.mean(row_cos(val_anchor, d_all[val_ids]))),
        "model_cos": float(np.mean(row_cos(val_pred, d_all[val_ids]))),
        "calibrated_cos": float(np.mean(row_cos(calibrated_val, d_all[val_ids]))),
        "path_cos": float(np.mean(row_cos(val_path, d_all[val_ids]))),
        "gate": summarize(val_gate),
    }
    print(json.dumps({"validation_metrics": validation_metrics, "alpha": alpha_by_regime}, indent=2), flush=True)

    if validation_metrics["calibrated_cos"] <= validation_metrics["anchor_cos"] + 1e-5:
        report = {
            "method": VERSION,
            "status": "analysis_only_no_candidate",
            "primary_hypothesis": "Official-H neural calibration can turn scalar point-cloud Sionna path fields into a better PDP predictor than the official KNN anchor.",
            "validation": validation,
            "validation_metrics": validation_metrics,
            "calibration": calibration,
            "local_gate": None,
            "outputs": {"report": PATHS.relative(REPORT), "zip": None},
        }
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return

    best_epoch = max(10, int(validation["best_epoch"]))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_anchor, final_w = make_anchor(
        train_pos[valid_ids], train_blocked[valid_ids], train_pos[valid_ids], train_blocked[valid_ids], d_all[valid_ids],
        query_ids=valid_ids, bank_ids=valid_ids,
    )
    test_anchor, test_w = make_anchor(
        test_pos, test_blocked, train_pos[valid_ids], train_blocked[valid_ids], d_all[valid_ids]
    )
    final_raw = field_features(fields, "train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids], final_anchor, final_w)
    test_raw = field_features(fields, "test", None, test_pos, test_blocked, test_anchor, test_w)
    final_features, [test_features], feature_stats = standardize(final_raw, [test_raw])
    seed_pred = []
    seed_path = []
    seed_gate = []
    final_training = []
    for seed in SEEDS:
        final_model, info = train_model(final_features, final_anchor, d_all[valid_ids], seed, best_epoch, device, None)
        torch.save(
            {"state_dict": final_model.state_dict(), "feature_stats": feature_stats, "seed": seed, "epochs": best_epoch},
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        pred, path_d, gate = predict(final_model, test_features, test_anchor, device)
        seed_pred.append(pred)
        seed_path.append(path_d)
        seed_gate.append(gate)
        final_training.append(info)
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_pred = normalize(np.mean(np.stack(seed_pred), axis=0))
    test_path = normalize(np.mean(np.stack(seed_path), axis=0))
    test_gate = np.mean(np.stack(seed_gate), axis=0).astype(np.float32)
    regimes = regime_of(test_blocked).astype(str)
    alpha = np.asarray([alpha_by_regime[regime] for regime in regimes], dtype=np.float32)
    current = v225.current_delay_marginals(BASE)
    target_d = normalize((1.0 - alpha[:, None]) * current["d"] + alpha[:, None] * test_pred)
    np.save(TARGET_OUT, target_d)
    projection = v225.apply_delay_targets(
        BASE,
        OUTPUT,
        {"d": target_d, "du": current["du"], "dh": current["dh"]},
        regimes,
        1.05,
    )
    score = v208.score_prediction(OUTPUT)
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        alpha=alpha,
        test_gate=test_gate,
        model_vs_anchor=row_cos(test_pred, test_anchor),
        path_vs_anchor=row_cos(test_path, test_anchor),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate" if score["score_traceback_leaked"] < LOCAL_GATE else "gate_passed_no_zip_yet",
        "primary_hypothesis": "Official-H neural calibration can turn scalar point-cloud Sionna path fields into a better PDP predictor than the official KNN anchor.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H PDP",
                "official train/test coordinates",
                "official point-cloud scene/path geometry",
                "scalar Sionna path fields as physics features",
                "neural calibration trained only on official train H",
                "strict v239 official-data carrier",
            ],
            "generation_does_not_use": [
                "Sionna CFR or MIMO RT H",
                "restored Etoile mesh/object/material labels",
                "synthetic channel labels",
                "v212/v217 teacher outputs",
                "local replay truth during generation or model selection",
                "hidden online labels",
            ],
            "local_truth_usage": "one final diagnostic score after output generation",
        },
        "training": {
            "device": str(device),
            "split_train_count": int(len(train_ids)),
            "validation_count": int(len(val_ids)),
            "best_epoch": best_epoch,
            "validation": validation,
            "validation_metrics": validation_metrics,
            "calibration": calibration,
            "alpha_by_regime": alpha_by_regime,
            "final_training": final_training,
        },
        "test_diagnostics": {
            "gate": summarize(test_gate),
            "alpha": summarize(alpha),
            "model_vs_anchor": summarize(row_cos(test_pred, test_anchor)),
            "path_vs_anchor": summarize(row_cos(test_path, test_anchor)),
        },
        "projection": {key: summarize(value) for key, value in projection.items()},
        "local_gate": {
            "required": LOCAL_GATE,
            "score": score,
            "passed": bool(score["score_traceback_leaked"] >= LOCAL_GATE),
            "gap": float(score["score_traceback_leaked"] - LOCAL_GATE),
        },
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
