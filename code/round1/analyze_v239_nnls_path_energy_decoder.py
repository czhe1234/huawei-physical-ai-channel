#!/usr/bin/env python3
"""v239: directly supervised path-energy decoder from official train PDP.

The v236/v237 path branch contributed only a few percent of the training
output, so its energy head received weak indirect supervision.  Here each
official train PDP is first decomposed over point-cloud path atoms with a
regularized non-negative least-squares solve.  A path Transformer then learns
the resulting per-path energy and offset distribution directly.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v233_official_joint_ad_manifold as v233
import analyze_v236_official_path_pdp_decoder as v236
import analyze_v238_raw_path_conditional_angle as v238
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v87_easy_field_projection import S_SC, summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 239
VERSION = "v239_nnls_path_energy_decoder"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v239_nnls_path_energy_decoder.json"
NNLS_CACHE = OUT_DIR / "v239_train_nnls_path_labels.npz"
TARGET_OUT = OUT_DIR / "v239_test_path_pdp.npy"
ACTION_OUT = OUT_DIR / "v239_nnls_path_energy_actions.npz"

BASE_V226 = v233.BASE
BASE_V237 = v238.SOURCE
FLOOR = 1e-10
GATE_SCORE = 0.645604
SHIFTS = np.asarray([-0.85, 0.0, 0.85], dtype=np.float32)
TAIL_COUNT = 6
TRAIN_EPOCHS = 220
BATCH_SIZE = 96
SEEDS = (23901, 23931, 23959)
REGIMES = ("clear", "light", "medium", "heavy")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_d(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=1, keepdims=True), TINY)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a).reshape(len(a), -1).astype(np.float64)
    y = np.asarray(b).reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def numpy_atoms(delay: np.ndarray, path_type: np.ndarray, valid: np.ndarray) -> np.ndarray:
    bins = np.arange(S_SC, dtype=np.float32)[None, None, None, :]
    sigma = np.where(path_type == 0, 0.62, 0.96).astype(np.float32)
    center = delay[:, :, None, None] + SHIFTS[None, None, :, None]
    path_atoms = np.exp(-0.5 * ((bins - center) / sigma[:, :, None, None]) ** 2).astype(np.float32)
    path_atoms *= valid[:, :, None, None].astype(np.float32)
    path_atoms /= np.maximum(path_atoms.sum(axis=3, keepdims=True), 1e-30)
    path_atoms = path_atoms.reshape(len(delay), -1, S_SC)

    x = np.arange(S_SC, dtype=np.float32)[None, :]
    early0 = np.exp(-0.5 * (x / 0.55) ** 2)
    early1 = np.exp(-0.5 * ((x - 1.0) / 0.75) ** 2)
    early2 = np.exp(-0.5 * ((x - 2.0) / 1.05) ** 2)
    broad = np.exp(-0.5 * ((x - 10.0) / 4.5) ** 2)
    max_delay = np.max(np.where(valid, delay, 0.0), axis=1)[:, None]
    delta = np.maximum(x - max_delay, 0.0)
    causal = (x >= max_delay).astype(np.float32)
    tail8 = np.exp(-delta / 8.0) * causal
    tail22 = np.exp(-delta / 22.0) * causal
    fixed = np.stack(
        [
            np.repeat(early0, len(delay), axis=0),
            np.repeat(early1, len(delay), axis=0),
            np.repeat(early2, len(delay), axis=0),
            np.repeat(broad, len(delay), axis=0),
            tail8,
            tail22,
        ],
        axis=1,
    ).astype(np.float32)
    fixed /= np.maximum(fixed.sum(axis=2, keepdims=True), 1e-30)
    return np.concatenate([path_atoms, fixed], axis=1).astype(np.float32)


def build_nnls_labels(target_d: np.ndarray, valid_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if NNLS_CACHE.is_file():
        data = np.load(NNLS_CACHE)
        return data["weights"], data["reconstruction"], json.loads(str(data["report_json"].item()))
    cache = np.load(v215.PATH_CACHE)
    delay = cache["train_delay_bin"].astype(np.float32)
    path_type = cache["train_path_type"].astype(np.int64)
    valid = cache["train_valid"].astype(bool)
    atoms = numpy_atoms(delay, path_type, valid)
    atom_count = atoms.shape[1]
    weights = np.zeros((len(target_d), atom_count), dtype=np.float32)
    recon = np.zeros_like(target_d, dtype=np.float32)
    ridge = 0.022
    sum_penalty = 0.55
    eye = math.sqrt(ridge) * np.eye(atom_count, dtype=np.float64)
    sum_row = math.sqrt(sum_penalty) * np.ones((1, atom_count), dtype=np.float64)
    zeros = np.zeros(atom_count, dtype=np.float64)
    for count, idx in enumerate(valid_ids, start=1):
        matrix = atoms[idx].T.astype(np.float64)
        augmented = np.vstack([matrix, eye, sum_row])
        rhs = np.concatenate([target_d[idx].astype(np.float64), zeros, [math.sqrt(sum_penalty)]])
        weight, _ = nnls(augmented, rhs, maxiter=atom_count * 8)
        if weight.sum() <= 1e-20:
            weight[-1] = 1.0
        weight /= weight.sum()
        weights[idx] = weight.astype(np.float32)
        recon[idx] = normalize_d((weight[None, :] @ atoms[idx]).astype(np.float32))[0]
        if count % 100 == 0 or count == len(valid_ids):
            print(f"v239 NNLS {count}/{len(valid_ids)}", flush=True)
    oracle_cos = row_cos(recon[valid_ids], target_d[valid_ids])
    path_mass = weights[valid_ids, :-TAIL_COUNT].sum(axis=1)
    report = {
        "atom_count": int(atom_count),
        "path_atom_count": int(atom_count - TAIL_COUNT),
        "tail_atom_count": TAIL_COUNT,
        "ridge": ridge,
        "sum_penalty": sum_penalty,
        "oracle_cos": summarize(oracle_cos),
        "path_mass": summarize(path_mass),
    }
    np.savez_compressed(
        NNLS_CACHE,
        weights=weights,
        reconstruction=recon,
        report_json=np.asarray(json.dumps(report)),
    )
    return weights, recon, report


def torch_atoms(
    raw_delay: torch.Tensor,
    path_type: torch.Tensor,
    path_valid: torch.Tensor,
) -> torch.Tensor:
    bins = torch.arange(S_SC, dtype=torch.float32, device=raw_delay.device)[None, None, None, :]
    shifts = torch.tensor(SHIFTS, dtype=torch.float32, device=raw_delay.device)[None, None, :, None]
    sigma = torch.where(path_type == 1, 0.62, 0.96).float()
    center = raw_delay[:, :, None, None] + shifts
    path = torch.exp(-0.5 * ((bins - center) / sigma[:, :, None, None]) ** 2)
    path = path * path_valid[:, :, None, None]
    path = path / path.sum(dim=3, keepdim=True).clamp_min(1e-30)
    path = path.flatten(1, 2)

    x = torch.arange(S_SC, dtype=torch.float32, device=raw_delay.device)[None, :]
    max_delay = torch.max(raw_delay.masked_fill(~path_valid, -1e9), dim=1).values[:, None]
    delta = torch.clamp(x - max_delay, min=0.0)
    causal = (x >= max_delay).float()
    fixed = torch.stack(
        [
            torch.exp(-0.5 * (x / 0.55) ** 2).expand(len(raw_delay), -1),
            torch.exp(-0.5 * ((x - 1.0) / 0.75) ** 2).expand(len(raw_delay), -1),
            torch.exp(-0.5 * ((x - 2.0) / 1.05) ** 2).expand(len(raw_delay), -1),
            torch.exp(-0.5 * ((x - 10.0) / 4.5) ** 2).expand(len(raw_delay), -1),
            torch.exp(-delta / 8.0) * causal,
            torch.exp(-delta / 22.0) * causal,
        ],
        dim=1,
    )
    fixed = fixed / fixed.sum(dim=2, keepdim=True).clamp_min(1e-30)
    return torch.cat([path, fixed], dim=1)


class DirectPathEnergyDecoder(nn.Module):
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
        self.path_trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.SiLU())
        self.global_trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.SiLU())
        self.path_energy = nn.Linear(256, 1)
        self.shift_weight = nn.Linear(256, len(SHIFTS))
        self.tail_weight = nn.Linear(256, TAIL_COUNT)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        raw_delay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.global_proj(global_features).unsqueeze(1)
        path = self.path_proj(torch.cat([path_numeric, self.type_emb(path_type)], dim=2))
        tokens = torch.cat([query, path], dim=1)
        mask = torch.cat(
            [torch.zeros((len(path_valid), 1), dtype=torch.bool, device=path_valid.device), ~path_valid], dim=1
        )
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        global_latent = self.global_trunk(encoded[:, 0])
        path_latent = self.path_trunk(encoded[:, 1:])
        path_logits = self.path_energy(path_latent) + self.shift_weight(path_latent)
        path_logits = path_logits.masked_fill(~path_valid[..., None], -1e9).flatten(1)
        logits = torch.cat([path_logits, self.tail_weight(global_latent)], dim=1)
        weights = torch.softmax(logits, dim=1)
        atoms = torch_atoms(raw_delay, path_type, path_valid)
        pred = torch.sum(weights[..., None] * atoms, dim=1)
        pred /= pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return pred, weights


@dataclass
class DecoderOutput:
    d: np.ndarray
    weights: np.ndarray


@torch.no_grad()
def predict_model(
    model: DirectPathEnergyDecoder,
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    device: torch.device,
) -> DecoderOutput:
    model.eval()
    atom_count = raw_delay.shape[1] * len(SHIFTS) + TAIL_COUNT
    d = np.empty((len(raw_delay), S_SC), dtype=np.float32)
    weights = np.empty((len(raw_delay), atom_count), dtype=np.float32)
    for start in range(0, len(raw_delay), 256):
        end = min(start + 256, len(raw_delay))
        pred, weight = model(
            torch.from_numpy(features.global_features[start:end]).to(device),
            torch.from_numpy(features.path_numeric[start:end]).to(device),
            torch.from_numpy(features.path_type[start:end]).to(device),
            torch.from_numpy(features.path_valid[start:end]).to(device),
            torch.from_numpy(raw_delay[start:end]).to(device),
        )
        d[start:end] = pred.cpu().numpy()
        weights[start:end] = weight.cpu().numpy()
    return DecoderOutput(d, weights)


def decoder_loss(
    pred: torch.Tensor,
    weights: torch.Tensor,
    truth_d: torch.Tensor,
    truth_weights: torch.Tensor,
) -> torch.Tensor:
    cos = 1.0 - F.cosine_similarity(pred, truth_d, dim=1).mean()
    hellinger = 1.0 - torch.sum(torch.sqrt(pred.clamp_min(FLOOR) * truth_d.clamp_min(FLOOR)), dim=1).mean()
    cdf = torch.mean(torch.abs(torch.cumsum(pred, dim=1) - torch.cumsum(truth_d, dim=1)))
    label = (truth_weights + 0.002 / truth_weights.shape[1])
    label /= label.sum(dim=1, keepdim=True).clamp_min(1e-30)
    weight_kl = torch.sum(label * (torch.log(label.clamp_min(FLOOR)) - torch.log(weights.clamp_min(FLOOR))), dim=1).mean()
    log_shape = F.smooth_l1_loss(torch.log(pred.clamp_min(FLOOR)), torch.log(truth_d.clamp_min(FLOOR)), beta=1.0)
    return 0.48 * cos + 0.16 * hellinger + 0.11 * cdf + 0.20 * weight_kl + 0.05 * log_shape


def train_model(
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    target_d: np.ndarray,
    target_weights: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[v233.FeatureSet, np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[DirectPathEnergyDecoder, dict[str, object]]:
    set_seed(seed)
    model = DirectPathEnergyDecoder(features.global_features.shape[1], features.path_numeric.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.2e-4, weight_decay=2.4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(target_d))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            pred, weights = model(
                torch.from_numpy(features.global_features[ids]).to(device),
                torch.from_numpy(features.path_numeric[ids]).to(device),
                torch.from_numpy(features.path_type[ids]).to(device),
                torch.from_numpy(features.path_valid[ids]).to(device),
                torch.from_numpy(raw_delay[ids]).to(device),
            )
            loss = decoder_loss(
                pred,
                weights,
                torch.from_numpy(target_d[ids]).to(device),
                torch.from_numpy(target_weights[ids]).to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if validation is not None:
                val_features, val_delay, val_target, val_weights = validation
                val_out = predict_model(model, val_features, val_delay, device)
                model_cos = float(np.mean(row_cos(val_out.d, val_target)))
                label_cos = float(np.mean(row_cos(val_out.weights, val_weights)))
                item.update({"val_model_cos": model_cos, "val_weight_cos": label_cos})
                if model_cos > best_metric:
                    best_metric = model_cos
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v239 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_cos": None if not np.isfinite(best_metric) else float(best_metric),
        "history": history,
    }


def calibrate_alpha(anchor: np.ndarray, pred: np.ndarray, truth: np.ndarray, regimes: np.ndarray) -> tuple[dict[str, float], dict[str, object]]:
    grid = np.linspace(0.0, 1.25, 26, dtype=np.float32)
    selected: dict[str, float] = {}
    report: dict[str, object] = {}
    for regime in REGIMES:
        mask = regimes.astype(str) == regime
        scores = []
        for alpha in grid:
            blend = normalize_d((1.0 - alpha) * anchor[mask] + alpha * pred[mask])
            scores.append(float(np.mean(row_cos(blend, truth[mask]))))
        best = int(np.argmax(scores))
        selected[regime] = float(grid[best])
        report[regime] = {
            "count": int(np.sum(mask)),
            "alpha": float(grid[best]),
            "anchor_cos": float(scores[0]),
            "best_cos": float(scores[best]),
        }
    return selected, report


def run_variant(
    name: str,
    source: Path,
    predicted_d: np.ndarray,
    alpha: np.ndarray,
    regimes: np.ndarray,
    clip: float,
) -> dict[str, object]:
    current = v225.current_delay_marginals(source)
    target_d = normalize_d((1.0 - alpha[:, None]) * current["d"] + alpha[:, None] * predicted_d)
    targets = {"d": target_d, "du": current["du"], "dh": current["dh"]}
    output = OUT_DIR / f"v239_{name}_Round1_Test_Channel.npy"
    diag = v225.apply_delay_targets(source, output, targets, regimes, clip)
    return {
        "name": name,
        "source": PATHS.relative(source),
        "output": PATHS.relative(output),
        "score": v208.score_prediction(output),
        "alpha": summarize(alpha),
        "target_l1": summarize(v225.l1_rows(target_d, current["d"])),
        "diagnostics": {key: summarize(value) for key, value in diag.items()},
    }


def main() -> None:
    for path in [BASE_V226, BASE_V237, v215.PATH_CACHE, v238.GEOM_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_targets, _raw_targets, _log_power, valid_delay = v215.load_targets()
    target_all = normalize_d(clean_targets["d"])
    support_raw, valid_support, log_power = v238.build_raw_support()
    valid = valid_delay & valid_support
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    nnls_weights, nnls_recon, nnls_report = build_nnls_labels(target_all, valid_ids)

    support = np.load(v238.OUT_DIR / "v238_train_raw_angle_delay_normalized.npy", mmap_mode="r")
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v238.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    bundle = v238.build_bundle(support, valid_ids, log_power, train_pos, test_pos, train_blocked, test_blocked)
    train_ids = bundle["train_ids"]
    val_ids = bundle["val_ids"]

    train_anchor_d = normalize_d(np.asarray(bundle["train_anchor"]).sum(axis=(1, 2)))
    val_anchor_d = normalize_d(np.asarray(bundle["val_anchor"]).sum(axis=(1, 2)))
    train_raw = v236.augment_raw_features(
        v233.raw_features(
            "train", train_ids, train_pos[train_ids], train_blocked[train_ids], bundle["train_anchor"],
            bundle["train_anchor_power"], bundle["w_train"],
        ),
        train_anchor_d,
    )
    val_raw = v236.augment_raw_features(
        v233.raw_features(
            "train", val_ids, train_pos[val_ids], train_blocked[val_ids], bundle["val_anchor"],
            bundle["val_anchor_power"], bundle["w_val"],
        ),
        val_anchor_d,
    )
    train_features, [val_features], _ = v233.standardize_feature_sets(train_raw, [val_raw])
    train_delay = v236.raw_path_delay("train", train_ids)
    val_delay = v236.raw_path_delay("train", val_ids)
    model, validation_report = train_model(
        train_features,
        train_delay,
        target_all[train_ids],
        nnls_weights[train_ids],
        SEEDS[0],
        TRAIN_EPOCHS,
        device,
        (val_features, val_delay, target_all[val_ids], nnls_weights[val_ids]),
    )
    val_out = predict_model(model, val_features, val_delay, device)
    val_alpha, calibration_report = calibrate_alpha(
        val_anchor_d, val_out.d, target_all[val_ids], regime_of(train_blocked[val_ids])
    )
    validation_metrics = {
        "anchor_cos": float(np.mean(row_cos(val_anchor_d, target_all[val_ids]))),
        "nnls_oracle_cos": float(np.mean(row_cos(nnls_recon[val_ids], target_all[val_ids]))),
        "model_cos": float(np.mean(row_cos(val_out.d, target_all[val_ids]))),
        "model_weight_cos": float(np.mean(row_cos(val_out.weights, nnls_weights[val_ids]))),
        "calibration": calibration_report,
    }
    best_epoch = max(10, int(validation_report["best_epoch"]))
    del model, val_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_anchor_d = normalize_d(np.asarray(bundle["final_anchor"]).sum(axis=(1, 2)))
    test_anchor_d = normalize_d(np.asarray(bundle["test_anchor"]).sum(axis=(1, 2)))
    final_raw = v236.augment_raw_features(
        v233.raw_features(
            "train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids], bundle["final_anchor"],
            bundle["final_anchor_power"], bundle["w_final"],
        ),
        final_anchor_d,
    )
    test_raw = v236.augment_raw_features(
        v233.raw_features(
            "test", None, test_pos, test_blocked, bundle["test_anchor"], bundle["test_anchor_power"], bundle["w_test"],
        ),
        test_anchor_d,
    )
    final_features, [test_features], final_stats = v233.standardize_feature_sets(final_raw, [test_raw])
    final_delay = v236.raw_path_delay("train", valid_ids)
    test_delay = v236.raw_path_delay("test", None)
    seed_outputs = []
    final_training = []
    for seed in SEEDS:
        final_model, info = train_model(
            final_features,
            final_delay,
            target_all[valid_ids],
            nnls_weights[valid_ids],
            seed,
            best_epoch,
            device,
            None,
        )
        torch.save(
            {"state_dict": final_model.state_dict(), "feature_stats": final_stats, "seed": seed, "epochs": best_epoch},
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        seed_outputs.append(predict_model(final_model, test_features, test_delay, device))
        final_training.append(info)
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_d = normalize_d(np.mean(np.stack([item.d for item in seed_outputs]), axis=0))
    test_weights = np.mean(np.stack([item.weights for item in seed_outputs]), axis=0).astype(np.float32)
    np.save(TARGET_OUT, test_d)
    agreement = np.mean(
        np.stack([
            row_cos(seed_outputs[i].d, seed_outputs[j].d)
            for i in range(len(seed_outputs)) for j in range(i + 1, len(seed_outputs))
        ]),
        axis=0,
    ).astype(np.float32)
    confidence = np.clip((agreement - 0.88) / 0.115, 0.0, 1.0).astype(np.float32)
    regimes = regime_of(test_blocked)
    alpha_base = np.asarray([val_alpha[str(regime)] for regime in regimes], dtype=np.float32)
    variants = [
        (
            "path_on_v226",
            BASE_V226,
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.05, "medium": 0.27, "heavy": 0.45}[str(r)] for r in regimes], dtype=np.float32)
                * confidence,
                0.0,
                0.49,
            ),
            1.04,
        ),
        (
            "calibrated_on_v226",
            BASE_V226,
            np.clip(alpha_base * (0.38 + 0.62 * confidence), 0.0, 0.92),
            1.04,
        ),
        (
            "path_on_v237",
            BASE_V237,
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.03, "medium": 0.16, "heavy": 0.28}[str(r)] for r in regimes], dtype=np.float32)
                * confidence,
                0.0,
                0.31,
            ),
            1.02,
        ),
        (
            "heavy_overdrive_on_v226",
            BASE_V226,
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.02, "medium": 0.34, "heavy": 0.62}[str(r)] for r in regimes], dtype=np.float32)
                * (0.44 + 0.56 * confidence),
                0.0,
                0.67,
            ),
            1.10,
        ),
    ]
    results = [run_variant(name, source, test_d, alpha.astype(np.float32), regimes, clip) for name, source, alpha, clip in variants]
    best = max(results, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        agreement=agreement,
        confidence=confidence,
        weights=test_weights,
        predicted_vs_anchor=row_cos(test_d, test_anchor_d),
        predicted_vs_v237=row_cos(test_d, v225.current_delay_marginals(BASE_V237)["d"]),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Direct NNLS supervision of per-path energy and delay offset fixes the weak indirect path-energy learning in v236/v237.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H PDP",
                "official train/test coordinates",
                "official point-cloud geometry and multibounce path candidates",
                "v226/v237 strict official-data carriers",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217/v221/v222 teacher outputs",
                "synthetic replay truth",
                "hidden online labels",
                "raw RT H",
            ],
        },
        "nnls": nnls_report,
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
            "predicted_vs_anchor": summarize(row_cos(test_d, test_anchor_d)),
            "predicted_vs_v237": summarize(row_cos(test_d, v225.current_delay_marginals(BASE_V237)["d"])),
            "predicted_path_mass": summarize(test_weights[:, :-TAIL_COUNT].sum(axis=1)),
        },
        "variants": results,
        "local_gate": {
            "required_score_traceback_leaked": GATE_SCORE,
            "best_variant": best["name"],
            "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
            "gap_to_gate": best["score"]["score_traceback_leaked"] - GATE_SCORE,
        },
        "outputs": {
            "nnls_labels": PATHS.relative(NNLS_CACHE),
            "target": PATHS.relative(TARGET_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
