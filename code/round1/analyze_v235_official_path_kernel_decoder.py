#!/usr/bin/env python3
"""v235 analysis: learnable official point-cloud path-kernel decoder.

Each public point-cloud path candidate predicts a separable horizontal-angle,
vertical-angle, and delay kernel plus an energy share.  A learned diffuse gate
retains the official train-H neighborhood anchor for propagation not explained
by the finite path set.  Training and generation use only released data and
official point-cloud-derived candidates.
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
import analyze_v233_official_joint_ad_manifold as v233
import generate_v215_pointcloud_physics_neural_field as v215
from generate_v87_easy_field_projection import MH, MV, S_SC, summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 235
VERSION = "v235_official_path_kernel_decoder"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v235_official_path_kernel_decoder.json"
TARGET_OUT = OUT_DIR / "v235_test_path_decoder_target.npy"
PATH_TARGET_OUT = OUT_DIR / "v235_test_path_only_target.npy"
ACTION_OUT = OUT_DIR / "v235_path_decoder_actions.npz"
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_EPOCHS = 180
BATCH_SIZE = 64
SEEDS = (23501, 23523, 23547)
FLOOR = 1e-10
GATE_SCORE = 0.645604
REGIMES = ("clear", "light", "medium", "heavy")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_path_delay(prefix: str, indices: np.ndarray | None) -> np.ndarray:
    cache = np.load(v215.PATH_CACHE)
    value = cache[f"{prefix}_delay_bin"].astype(np.float32)
    return value if indices is None else value[indices]


class PathKernelDecoder(nn.Module):
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
        self.path_trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.SiLU(), nn.Dropout(0.05))
        self.global_trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.SiLU())
        self.h_head = nn.Linear(256, MH)
        self.v_head = nn.Linear(256, MV)
        self.energy_head = nn.Linear(256, 1)
        self.offset_head = nn.Linear(256, 1)
        self.width_head = nn.Linear(256, 1)
        self.path_gate_head = nn.Linear(256, 1)
        nn.init.constant_(self.path_gate_head.bias, -1.5)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        raw_delay: torch.Tensor,
        anchor: torch.Tensor,
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

        h_prob = torch.softmax(self.h_head(path_latent), dim=2)
        v_prob = torch.softmax(self.v_head(path_latent), dim=2)
        center = raw_delay + 1.85 * torch.tanh(self.offset_head(path_latent).squeeze(2))
        width = 0.38 + 2.75 * torch.sigmoid(self.width_head(path_latent).squeeze(2))
        bins = torch.arange(S_SC, dtype=torch.float32, device=anchor.device)[None, None, :]
        delay_logits = -0.5 * ((bins - center[..., None]) / width[..., None]) ** 2
        delay_prob = torch.softmax(delay_logits, dim=2)

        path_logits = self.energy_head(path_latent).squeeze(2).masked_fill(~path_valid, -1e9)
        normalized_path_weight = torch.softmax(path_logits, dim=1)
        path_mass = 0.04 + 0.46 * torch.sigmoid(self.path_gate_head(global_latent).squeeze(1))
        path_weight = path_mass[:, None] * normalized_path_weight
        diffuse_weight = 1.0 - path_mass
        energy = torch.cat([diffuse_weight[:, None], path_weight], dim=1)
        path_support = torch.einsum("bp,bph,bpv,bpd->bhvd", path_weight, h_prob, v_prob, delay_prob)
        pure_path = path_support / path_mass[:, None, None, None].clamp_min(1e-8)
        pred = diffuse_weight[:, None, None, None] * anchor + path_support
        pred /= pred.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-30)
        return pred, pure_path, path_mass, energy


@dataclass
class DecoderOutput:
    support: np.ndarray
    path_support: np.ndarray
    path_mass: np.ndarray
    energy: np.ndarray


@torch.no_grad()
def predict_model(
    model: PathKernelDecoder,
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    anchor: np.ndarray,
    device: torch.device,
) -> DecoderOutput:
    model.eval()
    support = np.empty((len(anchor), MH, MV, S_SC), dtype=np.float32)
    path_support = np.empty_like(support)
    path_mass = np.empty(len(anchor), dtype=np.float32)
    energy = np.empty((len(anchor), raw_delay.shape[1] + 1), dtype=np.float32)
    for start in range(0, len(anchor), 96):
        end = min(start + 96, len(anchor))
        pred, pure, mass, weights = model(
            torch.from_numpy(features.global_features[start:end]).to(device),
            torch.from_numpy(features.path_numeric[start:end]).to(device),
            torch.from_numpy(features.path_type[start:end]).to(device),
            torch.from_numpy(features.path_valid[start:end]).to(device),
            torch.from_numpy(raw_delay[start:end]).to(device),
            torch.from_numpy(np.array(anchor[start:end], dtype=np.float32, copy=True)).to(device),
        )
        support[start:end] = pred.cpu().numpy()
        path_support[start:end] = pure.cpu().numpy()
        path_mass[start:end] = mass.cpu().numpy()
        energy[start:end] = weights.cpu().numpy()
    return DecoderOutput(support, path_support, path_mass, energy)


def decoder_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    path_mass: torch.Tensor,
    energy: torch.Tensor,
) -> torch.Tensor:
    p = pred.reshape(-1, MH, MV, S_SC)
    t = truth.reshape(-1, MH, MV, S_SC)
    joint = 1.0 - F.cosine_similarity(pred.reshape(len(pred), -1), truth.reshape(len(truth), -1), dim=1).mean()
    beam = 1.0 - F.cosine_similarity(p.sum(dim=3).reshape(len(p), -1), t.sum(dim=3).reshape(len(t), -1), dim=1).mean()
    delay = 1.0 - F.cosine_similarity(p.sum(dim=(1, 2)), t.sum(dim=(1, 2)), dim=1).mean()
    hellinger = 1.0 - torch.sum(torch.sqrt(pred.clamp_min(FLOOR) * truth.clamp_min(FLOOR)), dim=(1, 2, 3)).mean()
    energy_entropy = -torch.sum(energy * torch.log(energy.clamp_min(1e-12)), dim=1).mean()
    path_regularizer = torch.mean(torch.relu(path_mass - 0.78) ** 2)
    return 0.35 * joint + 0.24 * beam + 0.31 * delay + 0.07 * hellinger + 0.02 * path_regularizer + 0.001 * energy_entropy


def train_model(
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    anchor: np.ndarray,
    target: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[v233.FeatureSet, np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[PathKernelDecoder, dict[str, object]]:
    set_seed(seed)
    model = PathKernelDecoder(features.global_features.shape[1], features.path_numeric.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.2e-4, weight_decay=2.5e-4)
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
            anchor_t = torch.from_numpy(np.array(anchor[ids], dtype=np.float32, copy=True)).to(device)
            pred, _pure, mass, energy = model(
                torch.from_numpy(features.global_features[ids]).to(device),
                torch.from_numpy(features.path_numeric[ids]).to(device),
                torch.from_numpy(features.path_type[ids]).to(device),
                torch.from_numpy(features.path_valid[ids]).to(device),
                torch.from_numpy(raw_delay[ids]).to(device),
                anchor_t,
            )
            truth = torch.from_numpy(np.asarray(target[ids], dtype=np.float32)).to(device)
            loss = decoder_loss(pred, truth, mass, energy)
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
                metrics = v233.support_metrics(val_out.support, val_truth)
                anchor_metrics = v233.support_metrics(np.asarray(val_anchor), val_truth)
                item["val_model"] = metrics
                item["val_anchor"] = anchor_metrics
                item["val_path_mass"] = summarize(val_out.path_mass)
                if metrics["weighted"] > best_metric:
                    best_metric = metrics["weighted"]
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v235 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": None if not np.isfinite(best_metric) else float(best_metric),
        "history": history,
    }


def delay_only_target(current: np.ndarray, model: np.ndarray) -> np.ndarray:
    current_delay = np.maximum(current.sum(axis=(1, 2), keepdims=True), FLOOR)
    model_delay = np.maximum(model.sum(axis=(1, 2), keepdims=True), FLOOR)
    desired_delay = model_delay / np.maximum(model_delay.sum(axis=3, keepdims=True), TINY)
    conditional = current / current_delay
    return v233.normalize_support(conditional * desired_delay)


def angle_only_target(current: np.ndarray, model: np.ndarray) -> np.ndarray:
    model_delay = np.maximum(model.sum(axis=(1, 2), keepdims=True), FLOOR)
    conditional = model / model_delay
    current_delay = current.sum(axis=(1, 2), keepdims=True)
    return v233.normalize_support(conditional * current_delay)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support, valid, log_power = v233.load_normalized_train_support()
    valid_ids = np.flatnonzero(valid).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v233.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    bundle = v233.make_anchor_bundle(support, valid_ids, log_power, train_pos, test_pos, train_blocked, test_blocked)
    train_ids = bundle["train_ids"]
    val_ids = bundle["val_ids"]

    train_raw = v233.raw_features(
        "train", train_ids, train_pos[train_ids], train_blocked[train_ids], bundle["train_anchor"], bundle["train_anchor_power"], bundle["w_train"]
    )
    val_raw = v233.raw_features(
        "train", val_ids, train_pos[val_ids], train_blocked[val_ids], bundle["val_anchor"], bundle["val_anchor_power"], bundle["w_val"]
    )
    train_features, [val_features], _split_stats = v233.standardize_feature_sets(train_raw, [val_raw])
    train_delay = raw_path_delay("train", train_ids)
    val_delay = raw_path_delay("train", val_ids)
    train_target = np.asarray(support[train_ids], dtype=np.float32)
    val_target = np.asarray(support[val_ids], dtype=np.float32)
    validation_model, validation_report = train_model(
        train_features,
        train_delay,
        bundle["train_anchor"],
        train_target,
        SEEDS[0],
        TRAIN_EPOCHS,
        device,
        (val_features, val_delay, bundle["val_anchor"], val_target),
    )
    val_out = predict_model(validation_model, val_features, val_delay, bundle["val_anchor"], device)
    val_alpha, calibration_report = v233.calibrate_alpha(
        bundle["val_anchor"], val_out.support, val_target, regime_of(train_blocked[val_ids])
    )
    validation_metrics = {
        "anchor": v233.support_metrics(np.asarray(bundle["val_anchor"]), val_target),
        "model": v233.support_metrics(val_out.support, val_target),
        "path_only": v233.support_metrics(val_out.path_support, val_target),
        "path_mass": summarize(val_out.path_mass),
    }
    best_epoch = max(10, int(validation_report["best_epoch"]))
    del validation_model, val_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_raw = v233.raw_features(
        "train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids], bundle["final_anchor"], bundle["final_anchor_power"], bundle["w_final"]
    )
    test_raw = v233.raw_features(
        "test", None, test_pos, test_blocked, bundle["test_anchor"], bundle["test_anchor_power"], bundle["w_test"]
    )
    final_features, [test_features], final_stats = v233.standardize_feature_sets(final_raw, [test_raw])
    final_delay = raw_path_delay("train", valid_ids)
    test_delay = raw_path_delay("test", None)
    final_target = np.asarray(support[valid_ids], dtype=np.float32)
    seed_outputs = []
    final_training = []
    for seed in SEEDS:
        model, info = train_model(
            final_features,
            final_delay,
            bundle["final_anchor"],
            final_target,
            seed,
            best_epoch,
            device,
            None,
        )
        torch.save(
            {"state_dict": model.state_dict(), "feature_stats": final_stats, "seed": seed, "epochs": best_epoch},
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        seed_outputs.append(predict_model(model, test_features, test_delay, bundle["test_anchor"], device))
        final_training.append(info)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_model = v233.normalize_support(np.mean(np.stack([item.support for item in seed_outputs]), axis=0))
    test_path = v233.normalize_support(np.mean(np.stack([item.path_support for item in seed_outputs]), axis=0))
    path_mass = np.mean(np.stack([item.path_mass for item in seed_outputs]), axis=0).astype(np.float32)
    energy = np.mean(np.stack([item.energy for item in seed_outputs]), axis=0).astype(np.float32)
    np.save(TARGET_OUT, test_model)
    np.save(PATH_TARGET_OUT, test_path)

    current = np.asarray(v233.extract_base_support(test_pos), dtype=np.float32)
    regimes = regime_of(test_blocked)
    alpha_base = np.asarray([val_alpha[str(regime)] for regime in regimes], dtype=np.float32)
    seed_agreement = np.mean(
        np.stack(
            [
                v233.row_cos(seed_outputs[i].support, seed_outputs[j].support)
                for i in range(len(seed_outputs))
                for j in range(i + 1, len(seed_outputs))
            ]
        ),
        axis=0,
    ).astype(np.float32)
    confidence = np.clip((seed_agreement - 0.88) / 0.11, 0.0, 1.0).astype(np.float32)
    confidence *= np.clip((path_mass - 0.04) / 0.32, 0.0, 1.0).astype(np.float32)
    target_delay = delay_only_target(current, test_model)
    target_angle = angle_only_target(current, test_model)
    variants = [
        ("joint_linear", test_model, np.clip(alpha_base * (0.42 + 0.58 * confidence), 0.0, 0.88), "linear"),
        ("joint_log_aggressive", test_model, np.clip(1.35 * alpha_base * (0.44 + 0.56 * confidence), 0.0, 1.02), "log"),
        (
            "path_only_selective",
            test_path,
            np.clip(np.asarray([{"clear": 0.02, "light": 0.05, "medium": 0.18, "heavy": 0.28}[str(r)] for r in regimes], dtype=np.float32) * confidence, 0.0, 0.30),
            "linear",
        ),
        (
            "decoder_delay_only",
            target_delay,
            np.clip(np.asarray([{"clear": 0.0, "light": 0.05, "medium": 0.28, "heavy": 0.44}[str(r)] for r in regimes], dtype=np.float32) * (0.55 + 0.45 * confidence), 0.0, 0.48),
            "linear",
        ),
        (
            "decoder_angle_only",
            target_angle,
            np.clip(np.asarray([{"clear": 0.0, "light": 0.03, "medium": 0.16, "heavy": 0.24}[str(r)] for r in regimes], dtype=np.float32) * confidence, 0.0, 0.26),
            "linear",
        ),
    ]
    results = []
    for name, target, alpha, mode in variants:
        output, diagnostics = v233.apply_target(
            name,
            target,
            alpha.astype(np.float32),
            test_pos,
            mode,
            output_dir=OUT_DIR,
            version_prefix="v235",
        )
        results.append(
            {"name": name, "mode": mode, "output": PATHS.relative(output), "score": v208.score_prediction(output), "diagnostics": diagnostics}
        )
    best = max(results, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        path_mass=path_mass,
        energy=energy,
        seed_agreement=seed_agreement,
        confidence=confidence,
        model_vs_anchor=v233.row_cos(test_model, np.asarray(bundle["test_anchor"])),
        model_vs_v226=v233.row_cos(test_model, current),
        path_vs_v226=v233.row_cos(test_path, current),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Sionna-like propagation parameters are more transferable than whole support maps: learn per-path angle, delay, width, and energy kernels plus a diffuse anchor gate.",
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
            "seed_agreement": summarize(seed_agreement),
            "confidence": summarize(confidence),
            "model_vs_anchor": summarize(v233.row_cos(test_model, np.asarray(bundle["test_anchor"]))),
            "model_vs_v226": summarize(v233.row_cos(test_model, current)),
            "path_vs_v226": summarize(v233.row_cos(test_path, current)),
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
