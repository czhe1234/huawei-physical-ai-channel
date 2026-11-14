#!/usr/bin/env python3
"""v236 analysis: official point-cloud path PDP decoder.

The model learns only the 192-bin normalized PDP from released train H.  Each
official point-cloud path predicts delay offset, width, and energy; a separate
late-tail branch models unresolved diffuse energy.  Deployment changes only D
on the v226 carrier and preserves U-D/H-D allocation, angle structure, phase,
and total power.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v233_official_joint_ad_manifold as v233
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v87_easy_field_projection import S_SC, summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 236
VERSION = "v236_official_path_pdp_decoder"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v236_official_path_pdp_decoder.json"
TARGET_OUT = OUT_DIR / "v236_test_pdp_target.npy"
PATH_TARGET_OUT = OUT_DIR / "v236_test_path_pdp.npy"
ACTION_OUT = OUT_DIR / "v236_path_pdp_actions.npz"
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_EPOCHS = 180
BATCH_SIZE = 128
SEEDS = (23601, 23629, 23653)
FLOOR = 1e-10
GATE_SCORE = 0.645604
REGIMES = ("clear", "light", "medium", "heavy")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a.reshape(len(a), -1).astype(np.float64)
    y = b.reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def normalize_d(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=1, keepdims=True), TINY)
    return out.astype(np.float32)


def raw_path_delay(prefix: str, indices: np.ndarray | None) -> np.ndarray:
    cache = np.load(v215.PATH_CACHE)
    value = cache[f"{prefix}_delay_bin"].astype(np.float32)
    return value if indices is None else value[indices]


def augment_raw_features(
    raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    anchor_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global_features, path_numeric, path_type, path_valid = raw
    delay_features = np.concatenate(
        [np.log(np.maximum(anchor_d, FLOOR)), v215.d_stats(anchor_d)], axis=1
    ).astype(np.float32)
    return (
        np.concatenate([global_features, delay_features], axis=1).astype(np.float32),
        path_numeric,
        path_type,
        path_valid,
    )


class PathPDPDecoder(nn.Module):
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
        self.energy_head = nn.Linear(256, 1)
        self.offset_head = nn.Linear(256, 1)
        self.width_head = nn.Linear(256, 1)
        self.path_gate = nn.Linear(256, 1)
        self.tail_gate = nn.Linear(256, 1)
        self.tail_weight = nn.Linear(256, 4)
        nn.init.constant_(self.path_gate.bias, -1.2)
        nn.init.constant_(self.tail_gate.bias, -1.6)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        raw_delay: torch.Tensor,
        anchor_d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.global_proj(global_features).unsqueeze(1)
        path = self.path_proj(torch.cat([path_numeric, self.type_emb(path_type)], dim=2))
        tokens = torch.cat([query, path], dim=1)
        mask = torch.cat(
            [torch.zeros((len(path_valid), 1), dtype=torch.bool, device=path_valid.device), ~path_valid], dim=1
        )
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        global_latent = self.global_trunk(encoded[:, 0])
        path_latent = self.path_trunk(encoded[:, 1:])

        path_logits = self.energy_head(path_latent).squeeze(2).masked_fill(~path_valid, -1e9)
        path_weight = torch.softmax(path_logits, dim=1)
        center = raw_delay + 1.65 * torch.tanh(self.offset_head(path_latent).squeeze(2))
        width = 0.32 + 3.10 * torch.sigmoid(self.width_head(path_latent).squeeze(2))
        bins = torch.arange(S_SC, dtype=torch.float32, device=anchor_d.device)[None, None, :]
        path_kernel = torch.softmax(-0.5 * ((bins - center[..., None]) / width[..., None]) ** 2, dim=2)
        path_d = torch.sum(path_weight[..., None] * path_kernel, dim=1)

        max_delay = torch.max(raw_delay.masked_fill(~path_valid, -1e9), dim=1).values
        tail_bins = torch.arange(S_SC, dtype=torch.float32, device=anchor_d.device)[None, :]
        delta = torch.clamp(tail_bins - max_delay[:, None], min=0.0)
        causal = (tail_bins >= max_delay[:, None]).float()
        scales = torch.tensor([3.0, 7.0, 15.0, 30.0], dtype=torch.float32, device=anchor_d.device)
        tail_basis = torch.exp(-delta[:, None, :] / scales[None, :, None]) * causal[:, None, :]
        tail_basis /= tail_basis.sum(dim=2, keepdim=True).clamp_min(1e-30)
        tail_mix = torch.softmax(self.tail_weight(global_latent), dim=1)
        tail_d = torch.sum(tail_mix[..., None] * tail_basis, dim=1)

        path_mass = 0.03 + 0.52 * torch.sigmoid(self.path_gate(global_latent).squeeze(1))
        tail_mass = (1.0 - path_mass) * (0.02 + 0.34 * torch.sigmoid(self.tail_gate(global_latent).squeeze(1)))
        anchor_mass = 1.0 - path_mass - tail_mass
        pred = anchor_mass[:, None] * anchor_d + path_mass[:, None] * path_d + tail_mass[:, None] * tail_d
        pred /= pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return pred, path_d, tail_d, path_mass, tail_mass


@dataclass
class DecoderOutput:
    d: np.ndarray
    path_d: np.ndarray
    tail_d: np.ndarray
    path_mass: np.ndarray
    tail_mass: np.ndarray


@torch.no_grad()
def predict_model(
    model: PathPDPDecoder,
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    anchor_d: np.ndarray,
    device: torch.device,
) -> DecoderOutput:
    model.eval()
    d = np.empty_like(anchor_d, dtype=np.float32)
    path_d = np.empty_like(anchor_d, dtype=np.float32)
    tail_d = np.empty_like(anchor_d, dtype=np.float32)
    path_mass = np.empty(len(anchor_d), dtype=np.float32)
    tail_mass = np.empty(len(anchor_d), dtype=np.float32)
    for start in range(0, len(anchor_d), 256):
        end = min(start + 256, len(anchor_d))
        pred, path, tail, pmass, tmass = model(
            torch.from_numpy(features.global_features[start:end]).to(device),
            torch.from_numpy(features.path_numeric[start:end]).to(device),
            torch.from_numpy(features.path_type[start:end]).to(device),
            torch.from_numpy(features.path_valid[start:end]).to(device),
            torch.from_numpy(raw_delay[start:end]).to(device),
            torch.from_numpy(np.array(anchor_d[start:end], dtype=np.float32, copy=True)).to(device),
        )
        d[start:end] = pred.cpu().numpy()
        path_d[start:end] = path.cpu().numpy()
        tail_d[start:end] = tail.cpu().numpy()
        path_mass[start:end] = pmass.cpu().numpy()
        tail_mass[start:end] = tmass.cpu().numpy()
    return DecoderOutput(d, path_d, tail_d, path_mass, tail_mass)


def pdp_loss(pred: torch.Tensor, truth: torch.Tensor, path_mass: torch.Tensor, tail_mass: torch.Tensor) -> torch.Tensor:
    cos = 1.0 - F.cosine_similarity(pred, truth, dim=1).mean()
    hellinger = 1.0 - torch.sum(torch.sqrt(pred.clamp_min(FLOOR) * truth.clamp_min(FLOOR)), dim=1).mean()
    cdf = torch.mean(torch.abs(torch.cumsum(pred, dim=1) - torch.cumsum(truth, dim=1)))
    log_shape = F.smooth_l1_loss(torch.log(pred.clamp_min(FLOOR)), torch.log(truth.clamp_min(FLOOR)), beta=1.0)
    gate_reg = torch.mean(torch.relu(path_mass + tail_mass - 0.82) ** 2)
    return 0.58 * cos + 0.16 * hellinger + 0.13 * cdf + 0.12 * log_shape + 0.01 * gate_reg


def train_model(
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    anchor_d: np.ndarray,
    target_d: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[v233.FeatureSet, np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[PathPDPDecoder, dict[str, object]]:
    set_seed(seed)
    model = PathPDPDecoder(features.global_features.shape[1], features.path_numeric.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.5e-4, weight_decay=2.2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(anchor_d))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            pred, _path, _tail, pmass, tmass = model(
                torch.from_numpy(features.global_features[ids]).to(device),
                torch.from_numpy(features.path_numeric[ids]).to(device),
                torch.from_numpy(features.path_type[ids]).to(device),
                torch.from_numpy(features.path_valid[ids]).to(device),
                torch.from_numpy(raw_delay[ids]).to(device),
                torch.from_numpy(np.array(anchor_d[ids], dtype=np.float32, copy=True)).to(device),
            )
            truth = torch.from_numpy(target_d[ids]).to(device)
            loss = pdp_loss(pred, truth, pmass, tmass)
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
                val_out = predict_model(model, val_features, val_delay, val_anchor, device)
                model_cos = float(np.mean(row_cos(val_out.d, val_truth)))
                anchor_cos = float(np.mean(row_cos(val_anchor, val_truth)))
                item.update(
                    {
                        "val_model_cos": model_cos,
                        "val_anchor_cos": anchor_cos,
                        "val_path_cos": float(np.mean(row_cos(val_out.path_d, val_truth))),
                        "val_tail_cos": float(np.mean(row_cos(val_out.tail_d, val_truth))),
                        "val_path_mass": summarize(val_out.path_mass),
                        "val_tail_mass": summarize(val_out.tail_mass),
                    }
                )
                if model_cos > best_metric:
                    best_metric = model_cos
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v236 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
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
    selected = {}
    report = {}
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
    current: dict[str, np.ndarray],
    predicted_d: np.ndarray,
    alpha: np.ndarray,
    regimes: np.ndarray,
    clip: float,
) -> dict[str, object]:
    target_d = normalize_d((1.0 - alpha[:, None]) * current["d"] + alpha[:, None] * predicted_d)
    targets = {"d": target_d, "du": current["du"], "dh": current["dh"]}
    output = OUT_DIR / f"v236_{name}_Round1_Test_Channel.npy"
    diag = v225.apply_delay_targets(v233.BASE, output, targets, regimes, clip)
    return {
        "name": name,
        "output": PATHS.relative(output),
        "score": v208.score_prediction(output),
        "alpha": summarize(alpha),
        "target_l1": summarize(v225.l1_rows(target_d, current["d"])),
        "diagnostics": {key: summarize(value) for key, value in diag.items()},
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support, valid_joint, log_power = v233.load_normalized_train_support()
    clean_targets, _raw_targets, _clean_log_power, valid_delay = v215.load_targets()
    valid = valid_joint & valid_delay
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v233.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    bundle = v233.make_anchor_bundle(support, valid_ids, log_power, train_pos, test_pos, train_blocked, test_blocked)
    train_ids = bundle["train_ids"]
    val_ids = bundle["val_ids"]
    target_all = normalize_d(clean_targets["d"])
    idx_train, w_train = v218.neighbor_index_weights(
        train_pos[train_ids],
        train_blocked[train_ids],
        train_pos[train_ids],
        train_blocked[train_ids],
        query_ids=train_ids,
        bank_ids=train_ids,
        k=v233.ANCHOR_K,
    )
    idx_val, w_val = v218.neighbor_index_weights(
        train_pos[val_ids],
        train_blocked[val_ids],
        train_pos[train_ids],
        train_blocked[train_ids],
        k=v233.ANCHOR_K,
    )
    train_anchor_d = v218.weighted_average(target_all[train_ids], idx_train, w_train)
    val_anchor_d = v218.weighted_average(target_all[train_ids], idx_val, w_val)

    train_raw = augment_raw_features(v233.raw_features(
        "train", train_ids, train_pos[train_ids], train_blocked[train_ids], bundle["train_anchor"], bundle["train_anchor_power"], bundle["w_train"]
    ), train_anchor_d)
    val_raw = augment_raw_features(v233.raw_features(
        "train", val_ids, train_pos[val_ids], train_blocked[val_ids], bundle["val_anchor"], bundle["val_anchor_power"], bundle["w_val"]
    ), val_anchor_d)
    train_features, [val_features], _ = v233.standardize_feature_sets(train_raw, [val_raw])
    train_delay = raw_path_delay("train", train_ids)
    val_delay = raw_path_delay("train", val_ids)
    model, validation_report = train_model(
        train_features,
        train_delay,
        train_anchor_d,
        target_all[train_ids],
        SEEDS[0],
        TRAIN_EPOCHS,
        device,
        (val_features, val_delay, val_anchor_d, target_all[val_ids]),
    )
    val_out = predict_model(model, val_features, val_delay, val_anchor_d, device)
    val_alpha, calibration_report = calibrate_alpha(
        val_anchor_d, val_out.d, target_all[val_ids], regime_of(train_blocked[val_ids])
    )
    validation_metrics = {
        "anchor_cos": float(np.mean(row_cos(val_anchor_d, target_all[val_ids]))),
        "model_cos": float(np.mean(row_cos(val_out.d, target_all[val_ids]))),
        "path_cos": float(np.mean(row_cos(val_out.path_d, target_all[val_ids]))),
        "tail_cos": float(np.mean(row_cos(val_out.tail_d, target_all[val_ids]))),
        "path_mass": summarize(val_out.path_mass),
        "tail_mass": summarize(val_out.tail_mass),
    }
    best_epoch = max(10, int(validation_report["best_epoch"]))
    del model, val_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    idx_final, w_final = v218.neighbor_index_weights(
        train_pos[valid_ids],
        train_blocked[valid_ids],
        train_pos[valid_ids],
        train_blocked[valid_ids],
        query_ids=valid_ids,
        bank_ids=valid_ids,
        k=v233.ANCHOR_K,
    )
    idx_test, w_test = v218.neighbor_index_weights(
        test_pos,
        test_blocked,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        k=v233.ANCHOR_K,
    )
    final_anchor_d = v218.weighted_average(target_all[valid_ids], idx_final, w_final)
    test_anchor_d = v218.weighted_average(target_all[valid_ids], idx_test, w_test)
    final_raw = augment_raw_features(v233.raw_features(
        "train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids], bundle["final_anchor"], bundle["final_anchor_power"], bundle["w_final"]
    ), final_anchor_d)
    test_raw = augment_raw_features(v233.raw_features(
        "test", None, test_pos, test_blocked, bundle["test_anchor"], bundle["test_anchor_power"], bundle["w_test"]
    ), test_anchor_d)
    final_features, [test_features], final_stats = v233.standardize_feature_sets(final_raw, [test_raw])
    final_delay = raw_path_delay("train", valid_ids)
    test_delay = raw_path_delay("test", None)
    seed_outputs = []
    final_training = []
    for seed in SEEDS:
        final_model, info = train_model(
            final_features,
            final_delay,
            final_anchor_d,
            target_all[valid_ids],
            seed,
            best_epoch,
            device,
            None,
        )
        torch.save(
            {"state_dict": final_model.state_dict(), "feature_stats": final_stats, "seed": seed, "epochs": best_epoch},
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        seed_outputs.append(predict_model(final_model, test_features, test_delay, test_anchor_d, device))
        final_training.append(info)
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_d = normalize_d(np.mean(np.stack([item.d for item in seed_outputs]), axis=0))
    test_path_d = normalize_d(np.mean(np.stack([item.path_d for item in seed_outputs]), axis=0))
    test_tail_d = normalize_d(np.mean(np.stack([item.tail_d for item in seed_outputs]), axis=0))
    path_mass = np.mean(np.stack([item.path_mass for item in seed_outputs]), axis=0).astype(np.float32)
    tail_mass = np.mean(np.stack([item.tail_mass for item in seed_outputs]), axis=0).astype(np.float32)
    np.save(TARGET_OUT, test_d)
    np.save(PATH_TARGET_OUT, test_path_d)

    current = v225.current_delay_marginals(v233.BASE)
    regimes = regime_of(test_blocked)
    alpha_base = np.asarray([val_alpha[str(regime)] for regime in regimes], dtype=np.float32)
    agreement = np.mean(
        np.stack(
            [
                row_cos(seed_outputs[i].d, seed_outputs[j].d)
                for i in range(len(seed_outputs))
                for j in range(i + 1, len(seed_outputs))
            ]
        ),
        axis=0,
    ).astype(np.float32)
    confidence = np.clip((agreement - 0.94) / 0.055, 0.0, 1.0).astype(np.float32)
    variants = [
        ("model_calibrated", test_d, np.clip(alpha_base * (0.42 + 0.58 * confidence), 0.0, 0.92), 1.0),
        ("model_aggressive", test_d, np.clip(1.35 * alpha_base * (0.45 + 0.55 * confidence), 0.0, 1.08), 1.08),
        (
            "path_selective",
            test_path_d,
            np.clip(np.asarray([{"clear": 0.0, "light": 0.05, "medium": 0.24, "heavy": 0.40}[str(r)] for r in regimes], dtype=np.float32) * confidence, 0.0, 0.44),
            1.02,
        ),
        (
            "tail_selective",
            test_tail_d,
            np.clip(np.asarray([{"clear": 0.0, "light": 0.03, "medium": 0.14, "heavy": 0.25}[str(r)] for r in regimes], dtype=np.float32) * confidence, 0.0, 0.28),
            1.0,
        ),
        (
            "model_heavy_only",
            test_d,
            np.clip(np.asarray([{"clear": 0.0, "light": 0.0, "medium": 0.18, "heavy": 0.52}[str(r)] for r in regimes], dtype=np.float32) * (0.50 + 0.50 * confidence), 0.0, 0.56),
            1.04,
        ),
    ]
    results = [run_variant(name, current, pred, alpha.astype(np.float32), regimes, clip) for name, pred, alpha, clip in variants]
    best = max(results, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        path_mass=path_mass,
        tail_mass=tail_mass,
        agreement=agreement,
        confidence=confidence,
        model_vs_anchor=row_cos(test_d, test_anchor_d),
        model_vs_current=row_cos(test_d, current["d"]),
        path_vs_current=row_cos(test_path_d, current["d"]),
        tail_vs_current=row_cos(test_tail_d, current["d"]),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "The dominant v226-to-v212 gap is PDP. Learn Sionna-like per-path delay energy and unresolved late tails while preserving all non-D structure.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H PDP",
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
        },
        "training": {
            "device": str(device),
            "valid_train_count": int(len(valid_ids)),
            "split_train_count": int(len(train_ids)),
            "validation_count": int(len(val_ids)),
            "best_epoch": best_epoch,
            "validation": validation_report,
            "validation_metrics": validation_metrics,
            "calibration": calibration_report,
            "alpha_by_regime": val_alpha,
            "final_seeds": list(SEEDS),
            "final_training": final_training,
        },
        "test_diagnostics": {
            "path_mass": summarize(path_mass),
            "tail_mass": summarize(tail_mass),
            "agreement": summarize(agreement),
            "confidence": summarize(confidence),
            "model_vs_anchor": summarize(row_cos(test_d, test_anchor_d)),
            "model_vs_current": summarize(row_cos(test_d, current["d"])),
            "path_vs_current": summarize(row_cos(test_path_d, current["d"])),
            "tail_vs_current": summarize(row_cos(test_tail_d, current["d"])),
        },
        "variants": results,
        "local_gate": {
            "required_score_traceback_leaked": GATE_SCORE,
            "best_variant": best["name"],
            "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
            "gap_to_gate": best["score"]["score_traceback_leaked"] - GATE_SCORE,
        },
        "outputs": {
            "model_target": PATHS.relative(TARGET_OUT),
            "path_target": PATHS.relative(PATH_TARGET_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
