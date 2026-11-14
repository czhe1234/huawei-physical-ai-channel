#!/usr/bin/env python3
"""v238: raw-CIR path-conditional angle decoder on the fixed v237 PDP.

Point-cloud path delays live in the raw propagation-delay frame.  Earlier joint
path decoders paired them with carrier-derotated support.  This experiment
builds raw-CIR angle-delay supervision from official train H, learns only
P(H,V | D, path, geometry), and preserves the v237 PDP exactly at deployment.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v233_official_joint_ad_manifold as v233
import analyze_v236_official_path_pdp_decoder as v236
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 238
VERSION = "v238_raw_path_conditional_angle"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v238_raw_path_conditional_angle.json"
RAW_SUPPORT = OUT_DIR / "v238_train_raw_angle_delay_support.npy"
TRAIN_ANCHOR = OUT_DIR / "v238_split_train_anchor.npy"
VAL_ANCHOR = OUT_DIR / "v238_split_val_anchor.npy"
FINAL_ANCHOR = OUT_DIR / "v238_final_train_anchor.npy"
TEST_ANCHOR = OUT_DIR / "v238_test_anchor.npy"
TARGET_OUT = OUT_DIR / "v238_test_conditional_angle_target.npy"
ACTION_OUT = OUT_DIR / "v238_raw_path_conditional_angle_actions.npz"

SOURCE = PATHS.cache_dir(237) / "v237_path_on_v226_Round1_Test_Channel.npy"
GEOM_CACHE = v233.GEOM_CACHE
FLOOR = 1e-10
GATE_SCORE = 0.645604
VAL_COUNT = 320
ANCHOR_K = 72
TRAIN_EPOCHS = 200
BATCH_SIZE = 48
SEEDS = (23801, 23829, 23857)
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
    x = np.asarray(a).reshape(len(a), -1).astype(np.float64)
    y = np.asarray(b).reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def build_raw_support() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_h = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    if not RAW_SUPPORT.is_file():
        out = np.lib.format.open_memmap(
            RAW_SUPPORT,
            mode="w+",
            dtype=np.float32,
            shape=(len(train_h), MH, MV, S_SC),
        )
        for start in range(0, len(train_h), 8):
            end = min(start + 8, len(train_h))
            h = np.asarray(train_h[start:end], dtype=np.complex64)
            cir = np.fft.ifft(h, axis=-1)
            arr = cir.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
            raw_ad = np.fft.fft2(arr, axes=(2, 3))
            coll = np.sum(np.abs(raw_ad) ** 2, axis=(1, 4), dtype=np.float64).astype(np.float32)
            out[start:end] = coll
            out.flush()
            print(f"v238 raw support {end}/{len(train_h)}", flush=True)
    raw = np.load(RAW_SUPPORT, mmap_mode="r")
    total = np.sum(np.asarray(raw), axis=(1, 2, 3), dtype=np.float64)
    valid = total > 1e-20
    return raw, valid, np.log(np.maximum(total, 1e-30)).astype(np.float32)


def weighted_average(
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
    for start in range(0, len(idx), 8):
        end = min(start + 8, len(idx))
        block = np.zeros((end - start, MH, MV, S_SC), dtype=np.float32)
        for rank in range(idx.shape[1]):
            block += weights[start:end, rank, None, None, None] * np.asarray(
                bank[idx[start:end, rank]], dtype=np.float32
            )
        out[start:end] = normalize_support(block)
        out.flush()
        print(f"v238 anchor {output_path.name} {end}/{len(idx)}", flush=True)
    return np.load(output_path, mmap_mode="r")


def build_bundle(
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
        train_pos[train_ids], train_blocked[train_ids], train_pos[train_ids], train_blocked[train_ids],
        query_ids=train_ids, bank_ids=train_ids, k=ANCHOR_K,
    )
    idx_val, w_val = v218.neighbor_index_weights(
        train_pos[val_ids], train_blocked[val_ids], train_pos[train_ids], train_blocked[train_ids], k=ANCHOR_K,
    )
    split_bank = support[train_ids]
    train_anchor = weighted_average(split_bank, idx_train, w_train, TRAIN_ANCHOR)
    val_anchor = weighted_average(split_bank, idx_val, w_val, VAL_ANCHOR)

    idx_final, w_final = v218.neighbor_index_weights(
        train_pos[valid_ids], train_blocked[valid_ids], train_pos[valid_ids], train_blocked[valid_ids],
        query_ids=valid_ids, bank_ids=valid_ids, k=ANCHOR_K,
    )
    idx_test, w_test = v218.neighbor_index_weights(
        test_pos, test_blocked, train_pos[valid_ids], train_blocked[valid_ids], k=ANCHOR_K,
    )
    final_bank = support[valid_ids]
    final_anchor = weighted_average(final_bank, idx_final, w_final, FINAL_ANCHOR)
    test_anchor = weighted_average(final_bank, idx_test, w_test, TEST_ANCHOR)
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_anchor": train_anchor,
        "val_anchor": val_anchor,
        "final_anchor": final_anchor,
        "test_anchor": test_anchor,
        "train_anchor_power": np.sum(log_power[train_ids][idx_train] * w_train, axis=1).astype(np.float32),
        "val_anchor_power": np.sum(log_power[train_ids][idx_val] * w_val, axis=1).astype(np.float32),
        "final_anchor_power": np.sum(log_power[valid_ids][idx_final] * w_final, axis=1).astype(np.float32),
        "test_anchor_power": np.sum(log_power[valid_ids][idx_test] * w_test, axis=1).astype(np.float32),
        "w_train": w_train,
        "w_val": w_val,
        "w_final": w_final,
        "w_test": w_test,
    }


def conditional_from_support(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = x.sum(dim=(1, 2)).clamp_min(FLOOR)
    q = x / d[:, None, None, :]
    q /= q.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
    return q, d / d.sum(dim=1, keepdim=True).clamp_min(1e-30)


class ConditionalPathAngleDecoder(nn.Module):
    def __init__(self, global_dim: int, path_dim: int):
        super().__init__()
        d_model = 192
        self.type_emb = nn.Embedding(4, 24)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, 448), nn.SiLU(), nn.LayerNorm(448), nn.Dropout(0.05), nn.Linear(448, d_model)
        )
        self.path_proj = nn.Sequential(
            nn.Linear(path_dim + 24, 288), nn.SiLU(), nn.LayerNorm(288), nn.Linear(288, d_model)
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
        self.path_trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 320), nn.SiLU())
        self.global_trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.SiLU())
        self.angle_delta = nn.Linear(320, MH * MV)
        self.energy_head = nn.Linear(320, 1)
        self.offset_head = nn.Linear(320, 1)
        self.width_head = nn.Linear(320, 1)
        self.gate_head = nn.Linear(256, 1)
        nn.init.zeros_(self.angle_delta.weight)
        nn.init.zeros_(self.angle_delta.bias)
        nn.init.constant_(self.gate_head.bias, -1.65)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        raw_delay: torch.Tensor,
        anchor: torch.Tensor,
        fixed_d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.global_proj(global_features).unsqueeze(1)
        path = self.path_proj(torch.cat([path_numeric, self.type_emb(path_type)], dim=2))
        tokens = torch.cat([query, path], dim=1)
        mask = torch.cat(
            [torch.zeros((len(path_valid), 1), dtype=torch.bool, device=path_valid.device), ~path_valid], dim=1
        )
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        global_latent = self.global_trunk(encoded[:, 0])
        path_latent = self.path_trunk(encoded[:, 1:])

        anchor_q, _anchor_d = conditional_from_support(anchor)
        bins = torch.arange(S_SC, dtype=torch.float32, device=anchor.device)[None, None, :]
        center = raw_delay + 1.85 * torch.tanh(self.offset_head(path_latent).squeeze(2))
        width = 0.55 + 3.00 * torch.sigmoid(self.width_head(path_latent).squeeze(2))
        delay_kernel = torch.exp(-0.5 * ((bins - center[..., None]) / width[..., None]) ** 2)
        delay_kernel = delay_kernel * path_valid[..., None]

        anchor_local = torch.einsum("bhvd,bpd->bphv", anchor_q, delay_kernel)
        anchor_local /= anchor_local.sum(dim=(2, 3), keepdim=True).clamp_min(1e-30)
        residual = 2.4 * torch.tanh(self.angle_delta(path_latent)).reshape(-1, raw_delay.shape[1], MH, MV)
        path_q = torch.softmax(torch.log(anchor_local.clamp_min(FLOOR)).flatten(2) + residual.flatten(2), dim=2)
        path_q = path_q.reshape(-1, raw_delay.shape[1], MH, MV)

        energy = torch.exp(torch.clamp(self.energy_head(path_latent).squeeze(2), -4.0, 4.0)) * path_valid
        weighted_kernel = energy[..., None] * delay_kernel
        path_joint = torch.einsum("bpd,bphv->bhvd", weighted_kernel, path_q)
        path_cond = path_joint / path_joint.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        coverage = torch.amax(delay_kernel, dim=1).clamp(0.0, 1.0)
        global_gate = 0.03 + 0.72 * torch.sigmoid(self.gate_head(global_latent).squeeze(1))
        gate_d = global_gate[:, None] * coverage
        pred_q = (1.0 - gate_d[:, None, None, :]) * anchor_q + gate_d[:, None, None, :] * path_cond
        pred_q /= pred_q.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        pred = pred_q * fixed_d[:, None, None, :]
        pred /= pred.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-30)
        return pred, path_q, gate_d, center


@dataclass
class DecoderOutput:
    support: np.ndarray
    gate_d: np.ndarray
    center: np.ndarray


@torch.no_grad()
def predict_model(
    model: ConditionalPathAngleDecoder,
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    anchor: np.ndarray,
    fixed_d: np.ndarray,
    device: torch.device,
) -> DecoderOutput:
    model.eval()
    support = np.empty((len(anchor), MH, MV, S_SC), dtype=np.float32)
    gate_d = np.empty((len(anchor), S_SC), dtype=np.float32)
    center = np.empty_like(raw_delay, dtype=np.float32)
    for start in range(0, len(anchor), 64):
        end = min(start + 64, len(anchor))
        pred, _path_q, gate, ctr = model(
            torch.from_numpy(features.global_features[start:end]).to(device),
            torch.from_numpy(features.path_numeric[start:end]).to(device),
            torch.from_numpy(features.path_type[start:end]).to(device),
            torch.from_numpy(features.path_valid[start:end]).to(device),
            torch.from_numpy(raw_delay[start:end]).to(device),
            torch.from_numpy(np.array(anchor[start:end], dtype=np.float32, copy=True)).to(device),
            torch.from_numpy(np.array(fixed_d[start:end], dtype=np.float32, copy=True)).to(device),
        )
        support[start:end] = pred.cpu().numpy()
        gate_d[start:end] = gate.cpu().numpy()
        center[start:end] = ctr.cpu().numpy()
    return DecoderOutput(support, gate_d, center)


def angle_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred_beam = pred.sum(axis=3)
    truth_beam = truth.sum(axis=3)
    pred_d = np.maximum(pred.sum(axis=(1, 2)), FLOOR)
    truth_d = np.maximum(truth.sum(axis=(1, 2)), FLOOR)
    pred_q = pred / pred_d[:, None, None, :]
    truth_q = truth / truth_d[:, None, None, :]
    cond_bc = np.sum(np.sqrt(np.maximum(pred_q, 0.0) * np.maximum(truth_q, 0.0)), axis=(1, 2))
    return {
        "joint": float(np.mean(row_cos(pred, truth))),
        "beam": float(np.mean(row_cos(pred_beam, truth_beam))),
        "conditional_bc": float(np.mean(np.sum(truth_d * cond_bc, axis=1))),
        "weighted": float(
            0.42 * np.mean(row_cos(pred, truth))
            + 0.34 * np.mean(row_cos(pred_beam, truth_beam))
            + 0.24 * np.mean(np.sum(truth_d * cond_bc, axis=1))
        ),
    }


def angle_loss(pred: torch.Tensor, truth: torch.Tensor, gate_d: torch.Tensor) -> torch.Tensor:
    joint = 1.0 - F.cosine_similarity(pred.flatten(1), truth.flatten(1), dim=1).mean()
    beam = 1.0 - F.cosine_similarity(pred.sum(dim=3).flatten(1), truth.sum(dim=3).flatten(1), dim=1).mean()
    pred_q, _pred_d = conditional_from_support(pred)
    truth_q, truth_d = conditional_from_support(truth)
    bc = torch.sum(torch.sqrt(pred_q.clamp_min(FLOOR) * truth_q.clamp_min(FLOOR)), dim=(1, 2))
    conditional = 1.0 - torch.sum(truth_d * bc, dim=1).mean()
    entropy_reg = torch.mean(gate_d * (1.0 - gate_d))
    return 0.43 * joint + 0.34 * beam + 0.22 * conditional + 0.01 * entropy_reg


def train_model(
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    anchor: np.ndarray,
    target: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[v233.FeatureSet, np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[ConditionalPathAngleDecoder, dict[str, object]]:
    set_seed(seed)
    model = ConditionalPathAngleDecoder(features.global_features.shape[1], features.path_numeric.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.8e-4, weight_decay=2.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(anchor))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            truth = torch.from_numpy(np.asarray(target[ids], dtype=np.float32)).to(device)
            truth_d = truth.sum(dim=(1, 2))
            pred, _path_q, gate_d, _center = model(
                torch.from_numpy(features.global_features[ids]).to(device),
                torch.from_numpy(features.path_numeric[ids]).to(device),
                torch.from_numpy(features.path_type[ids]).to(device),
                torch.from_numpy(features.path_valid[ids]).to(device),
                torch.from_numpy(raw_delay[ids]).to(device),
                torch.from_numpy(np.array(anchor[ids], dtype=np.float32, copy=True)).to(device),
                truth_d,
            )
            loss = angle_loss(pred, truth, gate_d)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if validation is not None:
                val_features, val_delay, val_anchor, val_truth = validation
                val_truth_arr = np.asarray(val_truth, dtype=np.float32)
                val_d = val_truth_arr.sum(axis=(1, 2)).astype(np.float32)
                val_out = predict_model(model, val_features, val_delay, val_anchor, val_d, device)
                anchor_arr = np.asarray(val_anchor, dtype=np.float32)
                anchor_d = np.maximum(anchor_arr.sum(axis=(1, 2)), FLOOR)
                anchor_cond = anchor_arr / anchor_d[:, None, None, :]
                anchor_fixed = normalize_support(anchor_cond * val_d[:, None, None, :])
                model_metrics = angle_metrics(val_out.support, val_truth_arr)
                anchor_metrics = angle_metrics(anchor_fixed, val_truth_arr)
                item.update(
                    {
                        "val_model": model_metrics,
                        "val_anchor_fixed_d": anchor_metrics,
                        "val_gate": summarize(val_out.gate_d),
                    }
                )
                if model_metrics["weighted"] > best_metric:
                    best_metric = model_metrics["weighted"]
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v238 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
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
    model: np.ndarray,
    truth: np.ndarray,
    regimes: np.ndarray,
) -> tuple[dict[str, float], dict[str, object]]:
    grid = np.linspace(0.0, 1.20, 25, dtype=np.float32)
    selected: dict[str, float] = {}
    report: dict[str, object] = {}
    truth_d = truth.sum(axis=(1, 2))
    anchor_d = np.maximum(anchor.sum(axis=(1, 2)), FLOOR)
    model_d = np.maximum(model.sum(axis=(1, 2)), FLOOR)
    anchor_q = anchor / anchor_d[:, None, None, :]
    model_q = model / model_d[:, None, None, :]
    for regime in REGIMES:
        mask = regimes.astype(str) == regime
        scores = []
        for alpha in grid:
            q = (1.0 - alpha) * anchor_q[mask] + alpha * model_q[mask]
            q /= np.maximum(q.sum(axis=(1, 2), keepdims=True), TINY)
            support = normalize_support(q * truth_d[mask, None, None, :])
            scores.append(angle_metrics(support, truth[mask])["weighted"])
        best = int(np.argmax(scores))
        selected[regime] = float(grid[best])
        report[regime] = {
            "count": int(np.sum(mask)),
            "alpha": float(grid[best]),
            "anchor_score": float(scores[0]),
            "best_score": float(scores[best]),
        }
    return selected, report


def source_raw_support(path: Path) -> np.ndarray:
    source = np.load(path, mmap_mode="r")
    out = np.empty((len(source), MH, MV, S_SC), dtype=np.float32)
    for start in range(0, len(source), 8):
        end = min(start + 8, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        cir = np.fft.ifft(h, axis=-1)
        arr = cir.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
        raw_ad = np.fft.fft2(arr, axes=(2, 3))
        out[start:end] = normalize_support(
            np.sum(np.abs(raw_ad) ** 2, axis=(1, 4), dtype=np.float64).astype(np.float32)
        )
    return out


def apply_conditional_target(
    name: str,
    predicted: np.ndarray,
    alpha: np.ndarray,
    gain_clip: float,
) -> dict[str, object]:
    source = np.load(SOURCE, mmap_mode="r")
    output_path = OUT_DIR / f"v238_{name}_Round1_Test_Channel.npy"
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.complex64, shape=source.shape)
    diagnostics = {"support_l1": [], "d_l1": [], "power_ratio": [], "gain_max": []}
    for start in range(0, len(source), 8):
        end = min(start + 8, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        p0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        cir = np.fft.ifft(h0, axis=-1)
        arr = cir.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
        raw_ad = np.fft.fft2(arr, axes=(2, 3))
        power0 = np.abs(raw_ad).astype(np.float32) ** 2
        coll0 = np.sum(power0, axis=(1, 4), dtype=np.float64).astype(np.float32)
        total0 = np.maximum(coll0.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        current = coll0 / total0[:, None, None, None]
        current_d = np.maximum(current.sum(axis=(1, 2)), FLOOR)
        current_q = current / current_d[:, None, None, :]
        pred = np.asarray(predicted[start:end], dtype=np.float32)
        pred_d = np.maximum(pred.sum(axis=(1, 2)), FLOOR)
        pred_q = pred / pred_d[:, None, None, :]
        a = alpha[start:end, None, None, None]
        desired_q = (1.0 - a) * current_q + a * pred_q
        desired_q /= np.maximum(desired_q.sum(axis=(1, 2), keepdims=True), TINY)
        desired = desired_q * current_d[:, None, None, :]
        gain = np.sqrt(desired / np.maximum(current, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 1.0 / gain_clip, gain_clip)
        raw_ad1 = raw_ad * gain[:, None, :, :, None, :]

        coll1 = np.sum(np.abs(raw_ad1) ** 2, axis=(1, 4), dtype=np.float64).astype(np.float32)
        d1 = np.maximum(coll1.sum(axis=(1, 2)), 1e-30)
        d0_abs = np.maximum(coll0.sum(axis=(1, 2)), 1e-30)
        raw_ad1 *= np.sqrt(d0_abs / d1)[:, None, None, None, None, :].astype(np.float32)

        arr1 = np.fft.ifft2(raw_ad1, axes=(2, 3))
        h1 = np.fft.fft(arr1.reshape(end - start, MP * MH * MV, N_ANT, S_SC), axis=-1).astype(np.complex64)
        p1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        h1 *= np.sqrt(p0 / p1)[:, None, None, None].astype(np.float32)
        output[start:end] = h1
        output.flush()

        coll_after = np.sum(np.abs(raw_ad1) ** 2, axis=(1, 4), dtype=np.float64).astype(np.float32)
        support_after = coll_after / np.maximum(coll_after.sum(axis=(1, 2, 3), keepdims=True), TINY)
        diagnostics["support_l1"].append(np.sum(np.abs(support_after - current), axis=(1, 2, 3)).astype(np.float32))
        diagnostics["d_l1"].append(np.sum(np.abs(support_after.sum(axis=(1, 2)) - current_d), axis=1).astype(np.float32))
        diagnostics["power_ratio"].append((p1 / p0).astype(np.float32))
        diagnostics["gain_max"].append(np.max(gain.reshape(end - start, -1), axis=1).astype(np.float32))
        print(f"v238 apply {name} {end}/{len(source)}", flush=True)
    return {
        "name": name,
        "output": PATHS.relative(output_path),
        "score": v208.score_prediction(output_path),
        "alpha": summarize(alpha),
        "diagnostics": {key: summarize(np.concatenate(value)) for key, value in diagnostics.items()},
    }


def main() -> None:
    for path in [SOURCE, GEOM_CACHE, v215.PATH_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v238 device={device}", flush=True)
    support_raw, valid, log_power = build_raw_support()
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    support = np.lib.format.open_memmap(
        OUT_DIR / "v238_train_raw_angle_delay_normalized.npy",
        mode="w+" if not (OUT_DIR / "v238_train_raw_angle_delay_normalized.npy").is_file() else "r+",
        dtype=np.float32,
        shape=support_raw.shape,
    )
    if not np.any(support):
        for start in range(0, len(support_raw), 16):
            end = min(start + 16, len(support_raw))
            support[start:end] = normalize_support(np.asarray(support_raw[start:end], dtype=np.float32))
        support.flush()

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    bundle = build_bundle(support, valid_ids, log_power, train_pos, test_pos, train_blocked, test_blocked)
    train_ids = bundle["train_ids"]
    val_ids = bundle["val_ids"]

    train_raw = v233.raw_features(
        "train", train_ids, train_pos[train_ids], train_blocked[train_ids], bundle["train_anchor"],
        bundle["train_anchor_power"], bundle["w_train"],
    )
    val_raw = v233.raw_features(
        "train", val_ids, train_pos[val_ids], train_blocked[val_ids], bundle["val_anchor"],
        bundle["val_anchor_power"], bundle["w_val"],
    )
    train_features, [val_features], _ = v233.standardize_feature_sets(train_raw, [val_raw])
    train_delay = v236.raw_path_delay("train", train_ids)
    val_delay = v236.raw_path_delay("train", val_ids)
    train_target = np.asarray(support[train_ids], dtype=np.float32)
    val_target = np.asarray(support[val_ids], dtype=np.float32)
    model, validation_report = train_model(
        train_features, train_delay, bundle["train_anchor"], train_target, SEEDS[0], TRAIN_EPOCHS, device,
        (val_features, val_delay, bundle["val_anchor"], val_target),
    )
    val_d = val_target.sum(axis=(1, 2)).astype(np.float32)
    val_out = predict_model(model, val_features, val_delay, bundle["val_anchor"], val_d, device)
    val_alpha, calibration_report = calibrate_alpha(
        np.asarray(bundle["val_anchor"], dtype=np.float32), val_out.support, val_target,
        regime_of(train_blocked[val_ids]),
    )
    anchor_arr = np.asarray(bundle["val_anchor"], dtype=np.float32)
    anchor_q = anchor_arr / np.maximum(anchor_arr.sum(axis=(1, 2), keepdims=True), FLOOR)
    anchor_fixed = normalize_support(anchor_q * val_d[:, None, None, :])
    validation_metrics = {
        "anchor_fixed_d": angle_metrics(anchor_fixed, val_target),
        "model": angle_metrics(val_out.support, val_target),
        "gate_d": summarize(val_out.gate_d),
        "calibration": calibration_report,
    }
    best_epoch = max(10, int(validation_report["best_epoch"]))
    del model, val_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_raw = v233.raw_features(
        "train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids], bundle["final_anchor"],
        bundle["final_anchor_power"], bundle["w_final"],
    )
    test_raw = v233.raw_features(
        "test", None, test_pos, test_blocked, bundle["test_anchor"], bundle["test_anchor_power"], bundle["w_test"],
    )
    final_features, [test_features], final_stats = v233.standardize_feature_sets(final_raw, [test_raw])
    final_delay = v236.raw_path_delay("train", valid_ids)
    test_delay = v236.raw_path_delay("test", None)
    source_support = source_raw_support(SOURCE)
    source_d = source_support.sum(axis=(1, 2)).astype(np.float32)
    seed_outputs = []
    final_training = []
    final_target = np.asarray(support[valid_ids], dtype=np.float32)
    for seed in SEEDS:
        final_model, info = train_model(
            final_features, final_delay, bundle["final_anchor"], final_target, seed, best_epoch, device, None,
        )
        torch.save(
            {"state_dict": final_model.state_dict(), "feature_stats": final_stats, "seed": seed, "epochs": best_epoch},
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        seed_outputs.append(
            predict_model(final_model, test_features, test_delay, bundle["test_anchor"], source_d, device)
        )
        final_training.append(info)
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_target = normalize_support(np.mean(np.stack([item.support for item in seed_outputs]), axis=0))
    np.save(TARGET_OUT, test_target)
    agreement = np.mean(
        np.stack([
            row_cos(seed_outputs[i].support, seed_outputs[j].support)
            for i in range(len(seed_outputs)) for j in range(i + 1, len(seed_outputs))
        ]),
        axis=0,
    ).astype(np.float32)
    confidence = np.clip((agreement - 0.90) / 0.095, 0.0, 1.0).astype(np.float32)
    regimes = regime_of(test_blocked)
    alpha_base = np.asarray([val_alpha[str(regime)] for regime in regimes], dtype=np.float32)
    variants = [
        (
            "calibrated",
            np.clip(alpha_base * (0.35 + 0.65 * confidence), 0.0, 0.90),
            2.25,
        ),
        (
            "path_selective",
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.04, "medium": 0.18, "heavy": 0.32}[str(r)] for r in regimes], dtype=np.float32)
                * confidence,
                0.0,
                0.36,
            ),
            2.05,
        ),
        (
            "heavy_overdrive",
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.02, "medium": 0.26, "heavy": 0.52}[str(r)] for r in regimes], dtype=np.float32)
                * (0.45 + 0.55 * confidence),
                0.0,
                0.58,
            ),
            2.50,
        ),
    ]
    results = [apply_conditional_target(name, test_target, alpha.astype(np.float32), clip) for name, alpha, clip in variants]
    best = max(results, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        agreement=agreement,
        confidence=confidence,
        alpha_base=alpha_base,
        target_vs_source=row_cos(test_target, source_support),
        gate_d=np.mean(np.stack([item.gate_d for item in seed_outputs]), axis=0).astype(np.float32),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Raw point-cloud path delays must be paired with raw-CIR angle-delay support; conditional angular allocation can then improve PAS/leaked support while preserving the validated v237 PDP.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H",
                "official train/test coordinates",
                "official point-cloud geometry and multibounce path candidates",
                "v237 strict official-data carrier",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217/v221/v222 teacher outputs",
                "synthetic replay truth",
                "hidden online labels",
                "raw RT H",
            ],
        },
        "coordinate_alignment": {
            "path_delay_frame": "raw CIR",
            "training_support_frame": "raw CIR plus BS 2D FFT",
            "deployment_frame": "raw CIR plus BS 2D FFT",
            "preserved_target": "v237 per-point raw PDP",
        },
        "training": {
            "device": str(device),
            "valid_train_count": int(len(valid_ids)),
            "split_train_count": int(len(train_ids)),
            "validation_count": int(len(val_ids)),
            "best_epoch": best_epoch,
            "validation": validation_report,
            "validation_metrics": validation_metrics,
            "alpha_by_regime": val_alpha,
            "final_seeds": list(SEEDS),
            "final_training": final_training,
        },
        "test_diagnostics": {
            "agreement": summarize(agreement),
            "confidence": summarize(confidence),
            "target_vs_source": summarize(row_cos(test_target, source_support)),
        },
        "variants": results,
        "local_gate": {
            "required_score_traceback_leaked": GATE_SCORE,
            "best_variant": best["name"],
            "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
            "gap_to_gate": best["score"]["score_traceback_leaked"] - GATE_SCORE,
        },
        "outputs": {
            "target": PATHS.relative(TARGET_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
