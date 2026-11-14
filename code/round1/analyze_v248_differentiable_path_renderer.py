#!/usr/bin/env python3
"""v248: strict end-to-end differentiable point-cloud path renderer.

The renderer maps official point-cloud path candidates to the two dominant
evaluator spectra: PDP D and leaked S-H.  It learns path energy, sub-bin delay,
horizontal steering, path width, diffuse templates, and path/diffuse mixture
directly from official train H.  It never uses per-point NNLS labels or any
synthetic/Sionna channel target.
"""

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

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v245_neighbor_conditioned_spectral_operator as v245
import generate_v215_pointcloud_physics_neural_field as v215
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import MH, S_SC, summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 248
VERSION = "v248_differentiable_path_renderer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v248_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v248_path_renderer_targets.npz"
ACTIONS = OUT_DIR / "v248_path_renderer_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
PATH_CACHE = PATHS.shared_features / "multibounce_path_candidates.npz"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

LOCAL_GATE = 0.645604
VAL_COUNT = 320
TRAIN_EPOCHS = 260
BATCH_SIZE = 64
FINAL_SEEDS = (24801, 24819, 24847)
DIFFUSE_COMPONENTS = 8
REGIMES = ("clear", "light", "medium", "heavy")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def spectral_metrics(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


@dataclass
class RendererData:
    global_features: np.ndarray
    path_numeric: np.ndarray
    path_type: np.ndarray
    path_valid: np.ndarray
    delay: np.ndarray
    tx_dir: np.ndarray
    target_sh: np.ndarray
    target_d: np.ndarray


def load_path_raw(prefix: str, indices: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    cache = np.load(PATH_CACHE)
    delay = cache[f"{prefix}_delay_bin"].astype(np.float32)
    tx = cache[f"{prefix}_tx_dir"].astype(np.float32)
    if indices is not None:
        delay = delay[indices]
        tx = tx[indices]
    return np.nan_to_num(delay, nan=0.0), np.nan_to_num(tx, nan=0.0)


def make_raw_data(
    prefix: str,
    indices: np.ndarray | None,
    pos: np.ndarray,
    blocked: np.ndarray,
    target_sh: np.ndarray,
    target_d: np.ndarray,
) -> RendererData:
    global_features = v245.geometry_features(prefix, indices, pos, blocked)
    path_numeric, path_type, path_valid = v215.make_path_tokens(prefix, pos, indices)
    delay, tx = load_path_raw(prefix, indices)
    return RendererData(
        global_features=global_features,
        path_numeric=path_numeric.astype(np.float32),
        path_type=path_type.astype(np.int64),
        path_valid=path_valid.astype(bool),
        delay=delay.astype(np.float32),
        tx_dir=tx.astype(np.float32),
        target_sh=target_sh.astype(np.float32),
        target_d=target_d.astype(np.float32),
    )


def fit_feature_stats(data: RendererData) -> dict[str, np.ndarray]:
    valid_path = data.path_numeric[data.path_valid]
    return {
        "global_mean": data.global_features.mean(axis=0).astype(np.float32),
        "global_std": np.maximum(data.global_features.std(axis=0), 1e-5).astype(np.float32),
        "path_mean": valid_path.mean(axis=0).astype(np.float32),
        "path_std": np.maximum(valid_path.std(axis=0), 1e-5).astype(np.float32),
    }


def apply_feature_stats(data: RendererData, stats: dict[str, np.ndarray]) -> RendererData:
    global_features = np.clip(
        (data.global_features - stats["global_mean"]) / stats["global_std"], -8.0, 8.0
    ).astype(np.float32)
    path_numeric = np.clip(
        (data.path_numeric - stats["path_mean"][None, None, :])
        / stats["path_std"][None, None, :],
        -8.0,
        8.0,
    ).astype(np.float32)
    path_numeric[~data.path_valid] = 0.0
    return RendererData(
        global_features,
        path_numeric,
        data.path_type,
        data.path_valid,
        data.delay,
        data.tx_dir,
        data.target_sh,
        data.target_d,
    )


class DifferentiablePathRenderer(nn.Module):
    def __init__(self, global_dim: int, path_dim: int):
        super().__init__()
        width = 224
        self.type_embedding = nn.Embedding(4, 20)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 384),
            nn.SiLU(),
            nn.LayerNorm(384),
            nn.Linear(384, width),
        )
        self.path_encoder = nn.Sequential(
            nn.Linear(path_dim + 20, 320),
            nn.SiLU(),
            nn.LayerNorm(320),
            nn.Linear(320, width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=7,
            dim_feedforward=704,
            dropout=0.07,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.path_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 224), nn.SiLU(), nn.Linear(224, 10)
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 256), nn.SiLU(), nn.Linear(256, 2 + 2 * DIFFUSE_COMPONENTS)
        )
        self.orientation = nn.Parameter(torch.tensor(math.radians(-30.0), dtype=torch.float32))
        self.sh_delay_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        d_centers = torch.linspace(0.0, 22.0, DIFFUSE_COMPONENTS)
        d_bins = torch.arange(S_SC, dtype=torch.float32)[None, :]
        d_sigma = torch.linspace(1.2, 6.0, DIFFUSE_COMPONENTS)[:, None]
        d_init = -0.5 * ((d_bins - d_centers[:, None]) / d_sigma) ** 2
        self.diffuse_d_logits = nn.Parameter(d_init)

        s_centers = torch.linspace(170.0, 191.0, DIFFUSE_COMPONENTS)
        h_centers = torch.linspace(0.0, 15.0, DIFFUSE_COMPONENTS)
        s_bins = torch.arange(S_SC, dtype=torch.float32)[None, :, None]
        h_bins = torch.arange(MH, dtype=torch.float32)[None, None, :]
        ds = torch.remainder(s_bins - s_centers[:, None, None] + S_SC / 2, S_SC) - S_SC / 2
        dh = torch.remainder(h_bins - h_centers[:, None, None] + MH / 2, MH) - MH / 2
        sh_init = -0.5 * (ds / 5.0) ** 2 - 0.5 * (dh / 2.2) ** 2
        self.diffuse_sh_logits = nn.Parameter(sh_init)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        delay: torch.Tensor,
        tx_dir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        query = self.global_encoder(global_features).unsqueeze(1)
        path = self.path_encoder(
            torch.cat([path_numeric, self.type_embedding(path_type)], dim=2)
        )
        tokens = torch.cat([query, path], dim=1)
        mask = torch.cat(
            [
                torch.zeros((len(path_valid), 1), dtype=torch.bool, device=path_valid.device),
                ~path_valid,
            ],
            dim=1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=mask)
        global_params = self.global_head(encoded[:, 0])
        path_params = self.path_head(encoded[:, 1:])

        d_logits = path_params[..., 0].masked_fill(~path_valid, -1e9)
        d_shift = 2.6 * torch.tanh(path_params[..., 1])
        d_sigma = 0.38 + 2.8 * torch.sigmoid(path_params[..., 2])
        d_center = delay + d_shift
        d_bins = torch.arange(S_SC, dtype=torch.float32, device=delay.device)[None, None, :]
        d_kernel = torch.exp(-0.5 * ((d_bins - d_center[..., None]) / d_sigma[..., None]) ** 2)
        d_path_weights = torch.softmax(d_logits, dim=1)
        d_path = torch.sum(d_path_weights[..., None] * d_kernel, dim=1)
        d_path /= d_path.sum(dim=1, keepdim=True).clamp_min(1e-30)

        sh_logits = path_params[..., 3].masked_fill(~path_valid, -1e9)
        sh_s_shift = 5.0 * torch.tanh(path_params[..., 4])
        sh_s_sigma = 0.55 + 4.0 * torch.sigmoid(path_params[..., 5])
        h_correction = 2.8 * torch.tanh(path_params[..., 6])
        h_sigma = 0.42 + 2.8 * torch.sigmoid(path_params[..., 7])
        azimuth = torch.atan2(tx_dir[..., 1], tx_dir[..., 0]) - self.orientation
        h_center = torch.remainder((torch.sin(azimuth) + 1.0) * 0.5 * MH + h_correction, MH)
        sh_scale = 0.25 + 1.75 * torch.sigmoid(self.sh_delay_scale)
        s_center = torch.remainder(-sh_scale * delay + sh_s_shift, S_SC)
        s_bins = torch.arange(S_SC, dtype=torch.float32, device=delay.device)[None, None, :, None]
        h_bins = torch.arange(MH, dtype=torch.float32, device=delay.device)[None, None, None, :]
        ds = torch.remainder(s_bins - s_center[..., None, None] + S_SC / 2, S_SC) - S_SC / 2
        dh = torch.remainder(h_bins - h_center[..., None, None] + MH / 2, MH) - MH / 2
        sh_kernel = torch.exp(
            -0.5 * (ds / sh_s_sigma[..., None, None]) ** 2
            -0.5 * (dh / h_sigma[..., None, None]) ** 2
        )
        sh_path_weights = torch.softmax(sh_logits, dim=1)
        sh_path = torch.sum(sh_path_weights[..., None, None] * sh_kernel, dim=1)
        sh_path /= sh_path.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)

        d_diff_weights = torch.softmax(global_params[:, 2 : 2 + DIFFUSE_COMPONENTS], dim=1)
        sh_diff_weights = torch.softmax(global_params[:, 2 + DIFFUSE_COMPONENTS :], dim=1)
        d_templates = torch.softmax(self.diffuse_d_logits, dim=1)
        sh_templates = torch.softmax(self.diffuse_sh_logits.reshape(DIFFUSE_COMPONENTS, -1), dim=1).reshape(
            DIFFUSE_COMPONENTS, S_SC, MH
        )
        d_diff = d_diff_weights @ d_templates
        sh_diff = torch.einsum("bk,ksh->bsh", sh_diff_weights, sh_templates)
        d_path_mass = 0.12 + 0.86 * torch.sigmoid(global_params[:, 0])
        sh_path_mass = 0.12 + 0.86 * torch.sigmoid(global_params[:, 1])
        pred_d = d_path_mass[:, None] * d_path + (1.0 - d_path_mass[:, None]) * d_diff
        pred_sh = sh_path_mass[:, None, None] * sh_path + (1.0 - sh_path_mass[:, None, None]) * sh_diff
        pred_d /= pred_d.sum(dim=1, keepdim=True).clamp_min(1e-30)
        pred_sh /= pred_sh.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        diagnostics = {
            "d_path_mass": d_path_mass,
            "sh_path_mass": sh_path_mass,
            "d_path_weights": d_path_weights,
            "sh_path_weights": sh_path_weights,
            "d_shift": d_shift,
            "h_center": h_center,
            "sh_s_shift": sh_s_shift,
            "orientation_deg": torch.rad2deg(self.orientation).expand(len(pred_d)),
            "sh_delay_scale": sh_scale.expand(len(pred_d)),
        }
        return pred_sh, pred_d, diagnostics


def loss_function(
    pred_sh: torch.Tensor,
    pred_d: torch.Tensor,
    truth_sh: torch.Tensor,
    truth_d: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> torch.Tensor:
    sh_cos = 1.0 - F.cosine_similarity(pred_sh.flatten(1), truth_sh.flatten(1), dim=1).mean()
    d_cos = 1.0 - F.cosine_similarity(pred_d, truth_d, dim=1).mean()
    sh_s = 1.0 - F.cosine_similarity(pred_sh.sum(dim=2), truth_sh.sum(dim=2), dim=1).mean()
    sh_h = 1.0 - F.cosine_similarity(pred_sh.sum(dim=1), truth_sh.sum(dim=1), dim=1).mean()
    path_prior = 0.5 * (
        (diagnostics["d_path_mass"] - 0.72).abs().mean()
        + (diagnostics["sh_path_mass"] - 0.72).abs().mean()
    )
    return 0.42 * sh_cos + 0.38 * d_cos + 0.09 * sh_s + 0.08 * sh_h + 0.03 * path_prior


@torch.no_grad()
def predict(
    model: DifferentiablePathRenderer,
    data: RendererData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    sh_parts = []
    d_parts = []
    diagnostic_parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(data.global_features), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(data.global_features))
        sh, d, diagnostics = model(
            torch.from_numpy(data.global_features[start:end]).to(device),
            torch.from_numpy(data.path_numeric[start:end]).to(device),
            torch.from_numpy(data.path_type[start:end]).to(device),
            torch.from_numpy(data.path_valid[start:end]).to(device),
            torch.from_numpy(data.delay[start:end]).to(device),
            torch.from_numpy(data.tx_dir[start:end]).to(device),
        )
        sh_parts.append(sh.cpu().numpy())
        d_parts.append(d.cpu().numpy())
        for key, value in diagnostics.items():
            diagnostic_parts.setdefault(key, []).append(value.cpu().numpy())
    return (
        np.concatenate(sh_parts).astype(np.float32),
        np.concatenate(d_parts).astype(np.float32),
        {key: np.concatenate(value).astype(np.float32) for key, value in diagnostic_parts.items()},
    )


def train_model(
    data: RendererData,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: RendererData | None,
) -> tuple[DifferentiablePathRenderer, dict[str, object]]:
    set_seed(seed)
    model = DifferentiablePathRenderer(
        data.global_features.shape[1], data.path_numeric.shape[2]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.0e-4, weight_decay=2.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(data.global_features))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            pred_sh, pred_d, diagnostics = model(
                torch.from_numpy(data.global_features[ids]).to(device),
                torch.from_numpy(data.path_numeric[ids]).to(device),
                torch.from_numpy(data.path_type[ids]).to(device),
                torch.from_numpy(data.path_valid[ids]).to(device),
                torch.from_numpy(data.delay[ids]).to(device),
                torch.from_numpy(data.tx_dir[ids]).to(device),
            )
            loss = loss_function(
                pred_sh,
                pred_d,
                torch.from_numpy(data.target_sh[ids]).to(device),
                torch.from_numpy(data.target_d[ids]).to(device),
                diagnostics,
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
                val_sh, val_d, _ = predict(model, validation, device)
                val_metrics = spectral_metrics(
                    val_sh, val_d, validation.target_sh, validation.target_d
                )
                item["validation"] = val_metrics
                if val_metrics["weighted"] > best_metric:
                    best_metric = val_metrics["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
            print(f"v248 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": None if not np.isfinite(best_metric) else float(best_metric),
        "history": history,
    }


def main() -> None:
    for path in [TARGET_CACHE, PATH_CACHE, GEOM_CACHE, v247.BASE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_data = np.load(TARGET_CACHE)
    sh = target_data["sh"].astype(np.float32)
    d = target_data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    matched = choose_val_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, "matched-test", VERSION_NUMBER
    )
    val_ids = np.intersect1d(matched, valid_ids)
    fit_ids = np.setdiff1d(valid_ids, val_ids)

    fit_raw = make_raw_data(
        "train", fit_ids, train_pos[fit_ids], train_blocked[fit_ids], sh[fit_ids], d[fit_ids]
    )
    val_raw = make_raw_data(
        "train", val_ids, train_pos[val_ids], train_blocked[val_ids], sh[val_ids], d[val_ids]
    )
    stats = fit_feature_stats(fit_raw)
    fit_data = apply_feature_stats(fit_raw, stats)
    val_data = apply_feature_stats(val_raw, stats)
    validation_model, validation_training = train_model(
        fit_data, FINAL_SEEDS[0], TRAIN_EPOCHS, device, val_data
    )
    val_sh, val_d, val_diag = predict(validation_model, val_data, device)
    validation_metrics = spectral_metrics(val_sh, val_d, val_data.target_sh, val_data.target_d)
    selected_epochs = max(20, int(validation_training["best_epoch"]))
    del validation_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_raw = make_raw_data(
        "train",
        valid_ids,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        sh[valid_ids],
        d[valid_ids],
    )
    test_dummy_sh = np.zeros((len(test_pos), S_SC, MH), dtype=np.float32)
    test_dummy_d = np.zeros((len(test_pos), S_SC), dtype=np.float32)
    test_raw = make_raw_data(
        "test", None, test_pos, test_blocked, test_dummy_sh, test_dummy_d
    )
    final_stats = fit_feature_stats(final_raw)
    final_data = apply_feature_stats(final_raw, final_stats)
    test_data = apply_feature_stats(test_raw, final_stats)
    seed_sh = []
    seed_d = []
    seed_diag = []
    final_training = []
    for seed in FINAL_SEEDS:
        model, info = train_model(
            final_data, seed, selected_epochs, device, None
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
        pred_sh, pred_d, diagnostics = predict(model, test_data, device)
        seed_sh.append(pred_sh)
        seed_d.append(pred_d)
        seed_diag.append(diagnostics)
        final_training.append(info)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_sh = v247.normalize(np.mean(np.stack(seed_sh), axis=0))
    test_d = v247.normalize(np.mean(np.stack(seed_d), axis=0))
    mean_diag = {
        key: np.mean(np.stack([value[key] for value in seed_diag]), axis=0).astype(np.float32)
        for key in seed_diag[0]
    }
    np.savez_compressed(TARGETS, sh=test_sh, d=test_d, **mean_diag)
    projection_diag = v247.project(test_sh, test_d, output_path=OUTPUT)
    score = v208.score_prediction(OUTPUT)
    gate_passed = bool(float(score["score_traceback_leaked"]) >= LOCAL_GATE)
    np.savez_compressed(ACTIONS, **projection_diag)
    report = {
        "method": VERSION,
        "status": "gate_passed_needs_packaging" if gate_passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "End-to-end differentiable rendering of point-cloud path candidates can learn "
            "stable path energy/delay/angle structure that is lost when per-point NNLS labels "
            "are estimated first and regressed second."
        ),
        "compliance_boundary": {
            "generation_uses": [
                "official train H leaked S-H and PDP targets",
                "official train/test coordinates",
                "official point-cloud multibounce path candidates and geometry features",
                "v239 strict official-data carrier",
            ],
            "generation_does_not_use": [
                "Sionna or synthetic channel labels",
                "v212/v217/v221/v222 teacher outputs",
                "per-point synthetic or NNLS path labels",
                "local replay truth during training or selection",
                "hidden labels",
                "raw RT H",
            ],
            "local_truth_usage": "one final diagnostic score after generation",
        },
        "parameters": {
            "valid_train_count": int(len(valid_ids)),
            "fit_count": int(len(fit_ids)),
            "validation_count": int(len(val_ids)),
            "path_count": int(final_data.path_numeric.shape[1]),
            "diffuse_components": DIFFUSE_COMPONENTS,
            "selected_epochs": selected_epochs,
            "final_seeds": list(FINAL_SEEDS),
        },
        "validation": {
            "training": validation_training,
            "metrics": validation_metrics,
            "diagnostics": {key: summarize(value) for key, value in val_diag.items()},
        },
        "final_training": final_training,
        "test_renderer": {key: summarize(value) for key, value in mean_diag.items()},
        "projection": {key: summarize(value) for key, value in projection_diag.items()},
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
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
