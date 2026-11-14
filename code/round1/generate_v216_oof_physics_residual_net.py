#!/usr/bin/env python3
"""Generate v216: official-OOF physical residual neural net.

v212 is used only as architecture evidence: its online gain showed that
normalized PDP/delay support is the missing structure.  This script does not
train on v212-v209 test deltas.  It trains a neural residual model on official
train H:

* build a v209-style leave-one-out delay-support anchor on train points;
* describe each point with official geometry plus Sionna-rule summaries
  (multi-depth/frequency stability, long-delay shape and disagreement);
* learn anchor -> official train truth residual for D, U-D and H-D marginals;
* deploy the learned residual on the v209 test backbone with bounded support
  projection, preserving v209 phase texture and point total power.
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
import generate_v196_train_guided_delay_pdp_mouth as v196
import generate_v213_physical_constraint_action_model as v213
import generate_v215_pointcloud_physics_neural_field as v215
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import MH, MP, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v216_oof_physics_residual_net"
VERSION_NUMBER = 216

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v216_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v216_oof_physics_residual_actions.npz"
CHECKPOINT = PATHS.checkpoints / "v216_oof_physics_residual_net.pt"
REPORT = REPORT_DIR / "v216_oof_physics_residual_net_generate.json"
ZIP_OUT = PATHS.round1_versions / "v216_oof_physics_residual_net_candidate_pending.zip"

BASE = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"
V200_BASE = PATHS.cache_dir(200) / "v200_Round1_Test_Channel.npy"
V212_BASE = PATHS.cache_dir(212) / "v212_Round1_Test_Channel.npy"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
TRAIN_MARGINALS = PATHS.cache_dir(213) / "v213_train_sionna_constraint_marginals.npz"
TEST_MARGINALS = PATHS.cache_dir(213) / "v213_test_physical_constraint_marginals.npz"

FLOOR = 1e-12
VAL_COUNT = 300
SEEDS = (21601, 21617, 21633)
PCA_DIMS = {"d": 28, "du": 36, "dh": 44}
TRAIN_EPOCHS = 220
FINAL_EPOCHS_MAX = 220
BATCH_SIZE = 128
IPF_ITERS = 8
TEST_BATCH = 5


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


def l1_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(a.reshape(len(a), -1) - b.reshape(len(b), -1)), axis=1).astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a.reshape(len(a), -1).astype(np.float64)
    y = b.reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def delay_stats(x: np.ndarray) -> np.ndarray:
    arr = normalize(x, axis=x.ndim - 1)
    idx = np.arange(S_SC, dtype=np.float32)
    mean_bin = np.sum(arr * idx.reshape((1,) * (arr.ndim - 1) + (S_SC,)), axis=-1)
    var = np.sum(
        arr * (idx.reshape((1,) * (arr.ndim - 1) + (S_SC,)) - mean_bin[..., None]) ** 2,
        axis=-1,
    )
    entropy = -np.sum(arr * np.log(np.maximum(arr, FLOOR)), axis=-1) / math.log(S_SC)
    late = arr[..., 72:].sum(axis=-1)
    tail = arr[..., 112:].sum(axis=-1)
    top = np.max(arr, axis=-1)
    return np.concatenate(
        [
            (mean_bin / (S_SC - 1)).reshape(len(x), -1),
            (np.sqrt(np.maximum(var, 0.0)) / (S_SC - 1)).reshape(len(x), -1),
            entropy.reshape(len(x), -1),
            late.reshape(len(x), -1),
            tail.reshape(len(x), -1),
            top.reshape(len(x), -1),
        ],
        axis=1,
    ).astype(np.float32)


def position_features(pos: np.ndarray, blocked: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
    rel = pos.astype(np.float32) - bs[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arcsin(np.clip(rel[:, 2] / np.maximum(dist, 1e-6), -1.0, 1.0))
    onehot = np.stack(
        [(regimes.astype(str) == name).astype(np.float32) for name in ("clear", "light", "medium", "heavy")],
        axis=1,
    )
    xy = pos[:, :2].astype(np.float32)
    fourier = []
    for scale in (1.0, 2.0, 4.0, 8.0):
        fourier.extend(
            [
                np.sin(np.pi * scale * xy[:, 0] / 240.0),
                np.cos(np.pi * scale * xy[:, 0] / 240.0),
                np.sin(np.pi * scale * xy[:, 1] / 240.0),
                np.cos(np.pi * scale * xy[:, 1] / 240.0),
            ]
        )
    return np.concatenate(
        [
            pos.astype(np.float32) / np.asarray([260.0, 260.0, 60.0], dtype=np.float32),
            rel / 260.0,
            (dist / 260.0)[:, None],
            np.stack([np.sin(az), np.cos(az), np.sin(el), np.cos(el)], axis=1),
            blocked[:, None].astype(np.float32) / 160.0,
            onehot,
            np.stack(fourier, axis=1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def ratio_by_regime(
    truth: np.ndarray,
    ref: np.ndarray,
    regimes: np.ndarray,
    exponent: float,
    axes: tuple[int, ...] | int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for regime in ("clear", "light", "medium", "heavy"):
        mask = regimes.astype(str) == regime
        if not np.any(mask):
            out[regime] = np.ones(truth.shape[1:], dtype=np.float32)
            continue
        ratio = np.maximum(truth[mask].mean(axis=0), FLOOR) / np.maximum(ref[mask].mean(axis=0), FLOOR)
        out[regime] = np.power(np.clip(ratio, 0.18, 5.5), exponent).astype(np.float32)
    return out


def apply_ratio_by_regime(
    values: np.ndarray,
    ratios: dict[str, np.ndarray],
    regimes: np.ndarray,
    axes: tuple[int, ...] | int,
) -> np.ndarray:
    out = values.astype(np.float32).copy()
    for regime in ("clear", "light", "medium", "heavy"):
        mask = regimes.astype(str) == regime
        if np.any(mask):
            out[mask] *= ratios[regime]
    return normalize(out, axis=axes)


def candidate_blend(
    current: dict[str, np.ndarray],
    high: dict[str, np.ndarray],
    multi: dict[str, np.ndarray],
    anchor: dict[str, np.ndarray],
    high_total: np.ndarray,
    valid_count: np.ndarray,
    pair_l1: np.ndarray,
) -> dict[str, np.ndarray]:
    coverage = np.clip(valid_count.astype(np.float32) / 3.0, 0.0, 1.0)
    stability = np.clip(1.12 - 0.74 * pair_l1.astype(np.float32), 0.0, 1.0)
    nonzero = (high_total.astype(np.float32) > 1e-20).astype(np.float32)
    weight = np.clip(0.16 + 0.38 * coverage + 0.34 * stability + 0.12 * nonzero, 0.0, 1.0)
    result: dict[str, np.ndarray] = {}
    for key in ("d", "du", "dh"):
        axes = 1 if key == "d" else (1, 2)
        shape = (len(weight),) + (1,) * (current[key].ndim - 1)
        physical = normalize(0.72 * high[key] + 0.28 * multi[key], axis=axes)
        result[key] = normalize(
            weight.reshape(shape) * physical
            + 0.18 * anchor[key]
            + (0.82 - weight).reshape(shape) * current[key],
            axis=axes,
        )
    return result


def physical_features(
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    current: dict[str, np.ndarray],
    high: dict[str, np.ndarray],
    multi: dict[str, np.ndarray],
    anchor: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    high_total: np.ndarray,
    valid_count: np.ndarray,
    pair_l1: np.ndarray,
) -> np.ndarray:
    blocks: list[np.ndarray] = [
        position_features(pos, blocked, regimes),
        np.log1p(np.maximum(high_total.astype(np.float32), 0.0))[:, None],
        np.clip(valid_count.astype(np.float32), 0.0, 3.0)[:, None] / 3.0,
        pair_l1.astype(np.float32)[:, None],
    ]
    for family in (current, high, multi, anchor, candidate):
        blocks.extend([delay_stats(family["d"]), delay_stats(family["du"]), delay_stats(family["dh"])])
    distances = []
    for key in ("d", "du", "dh"):
        distances.extend(
            [
                l1_rows(high[key], current[key]),
                l1_rows(multi[key], current[key]),
                l1_rows(anchor[key], current[key]),
                l1_rows(candidate[key], current[key]),
                l1_rows(high[key], multi[key]),
                l1_rows(candidate[key], anchor[key]),
            ]
        )
    blocks.append(np.stack(distances, axis=1).astype(np.float32))
    return np.nan_to_num(np.concatenate(blocks, axis=1), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


@dataclass
class Prepared:
    x: np.ndarray
    current: dict[str, np.ndarray]
    target: dict[str, np.ndarray]
    coeff: dict[str, np.ndarray]
    regimes: np.ndarray


class ResidualNet(nn.Module):
    def __init__(self, input_dim: int, output_dims: dict[str, int]):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.SiLU(),
            nn.LayerNorm(384),
            nn.Dropout(0.08),
            nn.Linear(384, 384),
            nn.SiLU(),
            nn.LayerNorm(384),
            nn.Dropout(0.08),
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
        )
        self.heads = nn.ModuleDict({key: nn.Linear(256, dim) for key, dim in output_dims.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(x)
        return {key: head(z) for key, head in self.heads.items()}


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    mean = train_x.mean(axis=0).astype(np.float32)
    std = np.maximum(train_x.std(axis=0), 1e-5).astype(np.float32)

    def tr(x: np.ndarray) -> np.ndarray:
        return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)

    return tr(train_x), [tr(x) for x in others], {"mean": mean, "std": std}


def make_coeff(current: np.ndarray, target: np.ndarray, basis: v215.Basis) -> np.ndarray:
    residual = np.clip(
        np.log(np.maximum(target.reshape(len(target), -1), FLOOR))
        - np.log(np.maximum(current.reshape(len(current), -1), FLOOR)),
        -8.0,
        8.0,
    )
    raw = (residual - basis.mean[None, :]) @ basis.components.T
    return (raw / basis.coeff_scale[None, :]).astype(np.float32)


def torch_basis(basis: v215.Basis, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(basis.mean).to(device),
        torch.from_numpy(basis.components).to(device),
        torch.from_numpy(basis.coeff_scale).to(device),
    )


def reconstruct(
    coeff_norm: torch.Tensor,
    current: torch.Tensor,
    basis_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    mean, components, scale = basis_tensors
    delta = (coeff_norm * scale[None, :]) @ components + mean[None, :]
    logits = torch.log(current.reshape(len(current), -1).clamp_min(FLOOR)) + delta
    return torch.softmax(logits, dim=1)


def train_model(
    prepared: Prepared,
    bases: dict[str, v215.Basis],
    seed: int,
    epochs: int,
    device: torch.device,
    validation: Prepared | None = None,
) -> tuple[ResidualNet, dict[str, object]]:
    set_seed(seed)
    model = ResidualNet(prepared.x.shape[1], {key: len(bases[key].coeff_scale) for key in bases}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    basis_gpu = {key: torch_basis(basis, device) for key, basis in bases.items()}
    rng = np.random.default_rng(seed)
    n = len(prepared.x)
    history: list[dict[str, float]] = []
    best_score = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses_epoch = []
        for start in range(0, n, BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            pred_coeff = model(torch.from_numpy(prepared.x[ids]).to(device))
            loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            for key, weight in (("d", 0.52), ("du", 0.22), ("dh", 0.26)):
                current = torch.from_numpy(prepared.current[key][ids]).to(device)
                target = torch.from_numpy(prepared.target[key][ids]).to(device)
                coeff = torch.from_numpy(prepared.coeff[key][ids]).to(device)
                pred = reconstruct(pred_coeff[key], current, basis_gpu[key])
                cos = 1.0 - F.cosine_similarity(pred, target.reshape(len(ids), -1), dim=1).mean()
                coeff_loss = F.smooth_l1_loss(pred_coeff[key], coeff, beta=0.55)
                loss = loss + weight * (0.72 * cos + 0.28 * coeff_loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses_epoch.append(float(loss.detach().cpu()))
        sched.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item = {"epoch": float(epoch), "loss": float(np.mean(losses_epoch))}
            if validation is not None:
                pred = predict_model(model, validation, bases, device)
                score = (
                    0.52 * float(np.mean(row_cos(pred["d"], validation.target["d"])))
                    + 0.22 * float(np.mean(row_cos(pred["du"], validation.target["du"])))
                    + 0.26 * float(np.mean(row_cos(pred["dh"], validation.target["dh"])))
                )
                item["val_model_score"] = score
                item["val_anchor_score"] = (
                    0.52 * float(np.mean(row_cos(validation.current["d"], validation.target["d"])))
                    + 0.22 * float(np.mean(row_cos(validation.current["du"], validation.target["du"])))
                    + 0.26 * float(np.mean(row_cos(validation.current["dh"], validation.target["dh"])))
                )
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v216 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    best_value = None if not np.isfinite(best_score) else float(best_score)
    return model, {"seed": seed, "history": history, "best_epoch": int(best_epoch), "best_score": best_value}


@torch.no_grad()
def predict_model(
    model: ResidualNet,
    prepared: Prepared,
    bases: dict[str, v215.Basis],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    basis_gpu = {key: torch_basis(basis, device) for key, basis in bases.items()}
    out: dict[str, list[np.ndarray]] = {key: [] for key in bases}
    for start in range(0, len(prepared.x), 256):
        end = min(start + 256, len(prepared.x))
        coeff = model(torch.from_numpy(prepared.x[start:end]).to(device))
        for key, basis in bases.items():
            current = torch.from_numpy(prepared.current[key][start:end]).to(device)
            pred = reconstruct(coeff[key], current, basis_gpu[key]).cpu().numpy()
            out[key].append(pred.reshape((end - start,) + basis.shape).astype(np.float32))
    return {key: np.concatenate(value).astype(np.float32) for key, value in out.items()}


def calibrate(
    pred: dict[str, np.ndarray],
    val: Prepared,
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    alpha_grid = np.linspace(0.0, 1.35, 28, dtype=np.float32)
    support_alpha: dict[str, float] = {}
    action_cap: dict[str, float] = {}
    report: dict[str, object] = {}
    for regime in ("clear", "light", "medium", "heavy"):
        mask = val.regimes.astype(str) == regime
        if not np.any(mask):
            support_alpha[regime] = 0.0
            action_cap[regime] = 0.0
            continue
        scores = []
        for alpha in alpha_grid:
            score = 0.0
            for key, weight in (("d", 0.52), ("du", 0.22), ("dh", 0.26)):
                axes = 1 if key == "d" else (1, 2)
                blended = normalize((1.0 - alpha) * val.current[key][mask] + alpha * pred[key][mask], axis=axes)
                score += weight * float(np.mean(row_cos(blended, val.target[key][mask])))
            scores.append(score)
        best = int(np.argmax(scores))
        alpha = float(alpha_grid[best])
        # Keep the clean/light regimes restrained; heavy/medium are the expected
        # PDP/delay bottleneck from the v212 design evidence.
        cap_by_regime = {"clear": 0.0, "light": 0.18, "medium": 0.95, "heavy": 1.10}
        alpha = min(alpha, cap_by_regime[regime])
        support_alpha[regime] = alpha
        d_action = np.sum(
            np.abs(normalize((1.0 - alpha) * val.current["d"][mask] + alpha * pred["d"][mask], axis=1) - val.current["d"][mask]),
            axis=1,
        )
        action_cap[regime] = float(np.quantile(d_action, 0.88))
        report[regime] = {
            "count": int(np.sum(mask)),
            "alpha_raw": float(alpha_grid[best]),
            "alpha_deployed": alpha,
            "anchor_score": float(scores[0]),
            "best_score": float(scores[best]),
            "deployed_d_action_p88": action_cap[regime],
        }
    return support_alpha, action_cap, report


def build_data() -> tuple[Prepared, Prepared, Prepared, dict[str, v215.Basis], dict[str, object]]:
    train_m = v213.build_train_sionna_marginals()
    test_m = v213.build_test_marginals()
    geom = np.load(GEOM_CACHE)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    blocked_train = geom["train_blocked"].astype(np.float32)
    blocked_test = geom["test_blocked"].astype(np.float32)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)

    truth = {
        "d": normalize(train_m["truth_d"].astype(np.float32), axis=1),
        "du": normalize(train_m["truth_du"].astype(np.float32), axis=(1, 2)),
        "dh": normalize(train_m["truth_dh"].astype(np.float32), axis=(1, 2)),
        "valid": train_m["truth_valid"].astype(bool),
    }
    current_oof = v213.train_oof_anchor(truth, regimes_train, blocked_train)

    high_ratio = {
        "d": ratio_by_regime(truth["d"], train_m["high_d"], regimes_train, 0.44, axes=1),
        "du": ratio_by_regime(truth["du"], train_m["high_du"], regimes_train, 0.30, axes=(1, 2)),
        "dh": ratio_by_regime(truth["dh"], train_m["high_dh"], regimes_train, 0.28, axes=(1, 2)),
    }
    multi_ratio = {
        "d": ratio_by_regime(truth["d"], train_m["multi_d"], regimes_train, 0.44, axes=1),
        "du": ratio_by_regime(truth["du"], train_m["multi_du"], regimes_train, 0.30, axes=(1, 2)),
        "dh": ratio_by_regime(truth["dh"], train_m["multi_dh"], regimes_train, 0.28, axes=(1, 2)),
    }
    high_train = {
        "d": apply_ratio_by_regime(train_m["high_d"], high_ratio["d"], regimes_train, axes=1),
        "du": apply_ratio_by_regime(train_m["high_du"], high_ratio["du"], regimes_train, axes=(1, 2)),
        "dh": apply_ratio_by_regime(train_m["high_dh"], high_ratio["dh"], regimes_train, axes=(1, 2)),
    }
    multi_train = {
        "d": apply_ratio_by_regime(train_m["multi_d"], multi_ratio["d"], regimes_train, axes=1),
        "du": apply_ratio_by_regime(train_m["multi_du"], multi_ratio["du"], regimes_train, axes=(1, 2)),
        "dh": apply_ratio_by_regime(train_m["multi_dh"], multi_ratio["dh"], regimes_train, axes=(1, 2)),
    }
    high_test = {
        "d": apply_ratio_by_regime(test_m["high_d"], high_ratio["d"], regimes_test, axes=1),
        "du": apply_ratio_by_regime(test_m["high_du"], high_ratio["du"], regimes_test, axes=(1, 2)),
        "dh": apply_ratio_by_regime(test_m["high_dh"], high_ratio["dh"], regimes_test, axes=(1, 2)),
    }
    multi_test = {
        "d": apply_ratio_by_regime(test_m["multi_d"], multi_ratio["d"], regimes_test, axes=1),
        "du": apply_ratio_by_regime(test_m["multi_du"], multi_ratio["du"], regimes_test, axes=(1, 2)),
        "dh": apply_ratio_by_regime(test_m["multi_dh"], multi_ratio["dh"], regimes_test, axes=(1, 2)),
    }
    anchor_test = {
        "d": normalize(test_m["anchor_d"], axis=1),
        "du": normalize(test_m["anchor_du"], axis=(1, 2)),
        "dh": normalize(test_m["anchor_dh"], axis=(1, 2)),
    }
    current_test = {
        "d": normalize(test_m["current_d"], axis=1),
        "du": normalize(test_m["current_du"], axis=(1, 2)),
        "dh": normalize(test_m["current_dh"], axis=(1, 2)),
    }
    candidate_train = candidate_blend(
        current_oof, high_train, multi_train, current_oof,
        train_m["high_total_power"], train_m["multi_valid_count"], train_m["multi_pair_l1"],
    )
    candidate_test = candidate_blend(
        current_test, high_test, multi_test, anchor_test,
        test_m["high_total_power"], test_m["multi_valid_count"], test_m["multi_pair_l1"],
    )

    x_all_raw = physical_features(
        train_pos,
        blocked_train,
        regimes_train,
        current_oof,
        high_train,
        multi_train,
        current_oof,
        candidate_train,
        train_m["high_total_power"],
        train_m["multi_valid_count"],
        train_m["multi_pair_l1"],
    )
    x_test_raw = physical_features(
        test_pos,
        blocked_test,
        regimes_test,
        current_test,
        high_test,
        multi_test,
        anchor_test,
        candidate_test,
        test_m["high_total_power"],
        test_m["multi_valid_count"],
        test_m["multi_pair_l1"],
    )
    valid_ids = np.flatnonzero(truth["valid"])
    matched = choose_val_indices(train_pos[:, :2], test_pos[:, :2], VAL_COUNT, "matched-test", 216)
    val_ids = np.intersect1d(matched, valid_ids)
    train_ids = np.setdiff1d(valid_ids, val_ids)

    bases: dict[str, v215.Basis] = {}
    for key in ("d", "du", "dh"):
        basis, _ = v215.fit_basis(current_oof[key][train_ids], truth[key][train_ids], PCA_DIMS[key])
        bases[key] = basis
    x_train, (x_val, x_test), feature_stats = standardize(x_all_raw[train_ids], x_all_raw[val_ids], x_test_raw)

    train_current = {key: current_oof[key][train_ids] for key in ("d", "du", "dh")}
    val_current = {key: current_oof[key][val_ids] for key in ("d", "du", "dh")}
    train_target = {key: truth[key][train_ids] for key in ("d", "du", "dh")}
    val_target = {key: truth[key][val_ids] for key in ("d", "du", "dh")}
    train_coeff = {key: make_coeff(train_current[key], train_target[key], bases[key]) for key in ("d", "du", "dh")}
    val_coeff = {key: make_coeff(val_current[key], val_target[key], bases[key]) for key in ("d", "du", "dh")}
    train_prepared = Prepared(x_train, train_current, train_target, train_coeff, regimes_train[train_ids])
    val_prepared = Prepared(x_val, val_current, val_target, val_coeff, regimes_train[val_ids])

    final_bases: dict[str, v215.Basis] = {}
    for key in ("d", "du", "dh"):
        basis, _ = v215.fit_basis(current_oof[key][valid_ids], truth[key][valid_ids], PCA_DIMS[key])
        final_bases[key] = basis
    x_final, (x_test_final,), final_stats = standardize(x_all_raw[valid_ids], x_test_raw)
    final_current = {key: current_oof[key][valid_ids] for key in ("d", "du", "dh")}
    final_target = {key: truth[key][valid_ids] for key in ("d", "du", "dh")}
    final_coeff = {key: make_coeff(final_current[key], final_target[key], final_bases[key]) for key in ("d", "du", "dh")}
    final_prepared = Prepared(x_final, final_current, final_target, final_coeff, regimes_train[valid_ids])
    dummy_target = {key: current_test[key] for key in ("d", "du", "dh")}
    dummy_coeff = {key: np.zeros((len(test_pos), len(final_bases[key].coeff_scale)), dtype=np.float32) for key in ("d", "du", "dh")}
    test_prepared = Prepared(x_test_final, current_test, dummy_target, dummy_coeff, regimes_test)

    diagnostics = {
        "valid_train_count": int(len(valid_ids)),
        "validation_count": int(len(val_ids)),
        "feature_dim": int(x_all_raw.shape[1]),
        "feature_stats": {key: value for key, value in final_stats.items()},
        "train_anchor_scores": {
            key: summarize(row_cos(current_oof[key][valid_ids], truth[key][valid_ids])) for key in ("d", "du", "dh")
        },
        "train_candidate_scores": {
            key: summarize(row_cos(candidate_train[key][valid_ids], truth[key][valid_ids])) for key in ("d", "du", "dh")
        },
        "regimes_test": regimes_test,
        "current_test": current_test,
        "candidate_test": candidate_test,
    }
    return train_prepared, val_prepared, test_prepared, final_bases, {
        **diagnostics,
        "final_prepared": final_prepared,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "valid_ids": valid_ids,
    }


def apply_projection(
    prediction: dict[str, np.ndarray],
    current_test: dict[str, np.ndarray],
    regimes: np.ndarray,
    support_alpha: dict[str, float],
    action_cap: dict[str, float],
) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    diagnostics: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "full_l1_actual": [],
        "d_l1_actual": [],
        "du_l1_actual": [],
        "dh_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
    }
    alpha = np.asarray([support_alpha[str(r)] for r in regimes], dtype=np.float32)
    cap = np.asarray([action_cap[str(r)] for r in regimes], dtype=np.float32)
    raw_action = np.sum(np.abs(prediction["d"] - current_test["d"]), axis=1).astype(np.float32)
    deployed_alpha = alpha * np.minimum(1.0, cap / np.maximum(alpha * raw_action, 1e-8)).astype(np.float32)
    desired = {
        "d": normalize((1.0 - deployed_alpha[:, None]) * current_test["d"] + deployed_alpha[:, None] * prediction["d"], axis=1),
        "du": normalize(
            (1.0 - deployed_alpha[:, None, None]) * current_test["du"] + deployed_alpha[:, None, None] * prediction["du"],
            axis=(1, 2),
        ),
        "dh": normalize(
            (1.0 - deployed_alpha[:, None, None]) * current_test["dh"] + deployed_alpha[:, None, None] * prediction["dh"],
            axis=(1, 2),
        ),
    }
    for start in range(0, len(source), TEST_BATCH):
        end = min(start + TEST_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        cir0 = np.fft.ifft(h0, axis=-1)
        arr0 = cir0.reshape(end - start, MP, MH, 8, N_ANT, S_SC)
        power0 = (np.abs(arr0).astype(np.float32) ** 2).astype(np.float32)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        work = support0.copy()
        for _ in range(IPF_ITERS):
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired["d"][start:end], work.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)),
                "d",
                0.98,
            )
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired["du"][start:end], work.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)),
                "du",
                0.38,
            )
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired["dh"][start:end], work.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)),
                "dh",
                0.35,
            )
            work = normalize(work.reshape(end - start, -1), axis=1).reshape(support0.shape)
        gain = np.sqrt((work * total0[:, None, None, None, None, None]) / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.16, 6.2).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.42, 2.65)
        gain[strong1024] = np.clip(gain[strong1024], 0.58, 1.82)
        gain[strong128] = np.clip(gain[strong128], 0.82, 1.33)
        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.24 * (gain[local] - 1.0)
        h1 = np.fft.fft((arr0 * gain).reshape(end - start, 256, N_ANT, S_SC), axis=-1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()
        power1 = v196.delay_power_from_h(h1.astype(np.complex64))
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        diagnostics["actual_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diagnostics["full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diagnostics["d_l1_actual"].append(
            np.sum(
                np.abs(support1.sum(axis=(1, 2, 3, 4), dtype=np.float64) - support0.sum(axis=(1, 2, 3, 4), dtype=np.float64)),
                axis=1,
            ).astype(np.float32)
        )
        diagnostics["du_l1_actual"].append(
            np.sum(
                np.abs(support1.sum(axis=(1, 2, 3), dtype=np.float64) - support0.sum(axis=(1, 2, 3), dtype=np.float64)),
                axis=(1, 2),
            ).astype(np.float32)
        )
        diagnostics["dh_l1_actual"].append(
            np.sum(
                np.abs(support1.sum(axis=(1, 3, 4), dtype=np.float64) - support0.sum(axis=(1, 3, 4), dtype=np.float64)),
                axis=(1, 2),
            ).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diagnostics["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diagnostics["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diagnostics["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diagnostics["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v216 OOF residual projection {end}/{len(source)}", flush=True)
    result = {key: np.concatenate(value).astype(np.float32) for key, value in diagnostics.items()}
    result["deployed_alpha"] = deployed_alpha.astype(np.float32)
    result["desired_d"] = desired["d"].astype(np.float32)
    result["desired_du"] = desired["du"].astype(np.float32)
    result["desired_dh"] = desired["dh"].astype(np.float32)
    return result


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(CHECKPOINT, CHECKPOINT.name)
        for src in [
            PATHS.code / "generate_v216_oof_physics_residual_net.py",
            PATHS.code / "generate_v213_physical_constraint_action_model.py",
            PATHS.code / "generate_v215_pointcloud_physics_neural_field.py",
            PATHS.code / "generate_v209_adaptive_metric_projection_controller.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
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
    for path in [BASE, V200_BASE, V212_BASE, GEOM_CACHE]:
        require_file(path)
    # Build the v213 physical feature caches if they are missing.
    if not TRAIN_MARGINALS.is_file() or not TEST_MARGINALS.is_file():
        v213.build_train_sionna_marginals()
        v213.build_test_marginals()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_p, val_p, test_p, final_bases, data_diag = build_data()

    val_model, val_history = train_model(train_p, final_bases, SEEDS[0], TRAIN_EPOCHS, device, validation=val_p)
    val_pred = predict_model(val_model, val_p, final_bases, device)
    support_alpha, action_cap, calibration = calibrate(val_pred, val_p)
    selected_epochs = min(FINAL_EPOCHS_MAX, max(20, int(val_history["best_epoch"])))
    validation = {
        "anchor_scores": {
            key: summarize(row_cos(val_p.current[key], val_p.target[key])) for key in ("d", "du", "dh")
        },
        "model_scores": {
            key: summarize(row_cos(val_pred[key], val_p.target[key])) for key in ("d", "du", "dh")
        },
        "calibration": calibration,
        "history": val_history,
    }
    print(json.dumps(validation, indent=2, ensure_ascii=False), flush=True)
    del val_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_p: Prepared = data_diag["final_prepared"]  # type: ignore[assignment]
    test_predictions = []
    states = []
    histories = []
    for seed in SEEDS:
        model, history = train_model(final_p, final_bases, seed, selected_epochs, device)
        test_predictions.append(predict_model(model, test_p, final_bases, device))
        states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        histories.append(history)
    pred = {
        key: normalize(np.mean(np.stack([p[key] for p in test_predictions], axis=0), axis=0), axis=1 if key == "d" else (1, 2))
        for key in ("d", "du", "dh")
    }
    regimes_test = data_diag["regimes_test"].astype(str)  # type: ignore[index]
    current_test = data_diag["current_test"]  # type: ignore[assignment]
    apply_diag = apply_projection(pred, current_test, regimes_test, support_alpha, action_cap)
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes_test.astype("U8"),
        support_alpha=np.asarray([support_alpha[str(r)] for r in regimes_test], dtype=np.float32),
        support_action_cap=np.asarray([action_cap[str(r)] for r in regimes_test], dtype=np.float32),
        predicted_d=pred["d"],
        predicted_du=pred["du"],
        predicted_dh=pred["dh"],
        **apply_diag,
    )
    checkpoint = {
        "method": VERSION,
        "seeds": SEEDS,
        "state_dicts": states,
        "input_dim": int(final_p.x.shape[1]),
        "pca_dims": PCA_DIMS,
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
        "support_action_cap": action_cap,
        "selected_epochs": selected_epochs,
        "backbone": PATHS.relative(BASE),
        "neural_head_official_train_residual": True,
        "v212_role": "architecture evidence only; no v212-v209 test residual target",
        "full_pipeline_one_click_from_raw_inputs": False,
    }
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, CHECKPOINT)

    comparison_to_v209 = track.compare_output_to(OUTPUT, BASE, regimes_test)
    comparison_to_v200 = track.compare_output_to(OUTPUT, V200_BASE, regimes_test)
    comparison_to_v212 = track.compare_output_to(OUTPUT, V212_BASE, regimes_test)
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v212 should guide architecture, not labels.  A neural model trained on official train OOF residuals can "
            "learn when Sionna-rule features imply delayed PDP/support corrections and deploy that residual on v209."
        ),
        "compliance_boundary": {
            "neural_model": "official-train supervised MLP residual net for D/U-D/H-D support",
            "official_supervision": "released train H only; exact-zero rows excluded through v213 truth_valid",
            "sionna_role": "low-dimensional physical-rule features and stability summaries, not raw phase/power or v212-v209 labels",
            "excluded": "no v212 teacher residual target, no raw Sionna phase, no raw Sionna power, no full-H direct replacement",
            "decoder": "bounded support projection preserving v209 point total power and phase texture",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "selected_epochs": selected_epochs,
            "seeds": SEEDS,
            "pca_dims": PCA_DIMS,
            "train_epochs": TRAIN_EPOCHS,
            "support_alpha": support_alpha,
            "support_action_cap": action_cap,
            "ipf_iters": IPF_ITERS,
            "power_policy": "preserve v209 point total power exactly",
            "phase_policy": "preserve v209 delay-domain phase texture; amplitude-only support projection",
        },
        "data_diagnostics": {
            "valid_train_count": data_diag["valid_train_count"],
            "validation_count": data_diag["validation_count"],
            "feature_dim": data_diag["feature_dim"],
            "train_anchor_scores": data_diag["train_anchor_scores"],
            "train_candidate_scores": data_diag["train_candidate_scores"],
        },
        "validation": validation,
        "final_histories": histories,
        "action_counts": {
            "active": int(np.sum(apply_diag["deployed_alpha"] > 1e-7)),
            "active_by_regime": {
                regime: int(np.sum((apply_diag["deployed_alpha"] > 1e-7) & (regimes_test == regime)))
                for regime in ("clear", "light", "medium", "heavy")
            },
        },
        "diagnostics": {
            "overall": {
                key: summarize(value)
                for key, value in apply_diag.items()
                if key not in {"desired_d", "desired_du", "desired_dh"}
            },
            "by_regime": by_regime_summary(
                regimes_test,
                {key: value for key, value in apply_diag.items() if key not in {"desired_d", "desired_du", "desired_dh"}},
            ),
        },
        "comparison_to_v209": comparison_to_v209,
        "comparison_to_v200": comparison_to_v200,
        "comparison_to_v212_diagnostic_only": comparison_to_v212,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
