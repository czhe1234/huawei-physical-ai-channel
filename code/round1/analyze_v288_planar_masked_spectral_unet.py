#!/usr/bin/env python3
"""Whole-scene planar masked U-Net for official spectral-field inpainting."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
import analyze_v283_joint_geometry_path_renderer as v283
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 288
VERSION = "v288_planar_masked_spectral_unet"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v288_planar_field_targets.npz"
OUTPUT = OUT_DIR / "v288_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"

LOCAL_GATE = 0.645604
X_MIN = 46.0
X_MAX = 254.0
Y_MIN = -202.0
Y_MAX = 102.0
RESOLUTION = 2.0
GRID_W = int(round((X_MAX - X_MIN) / RESOLUTION)) + 1
GRID_H = int(round((Y_MAX - Y_MIN) / RESOLUTION)) + 1
PCA_COMPONENTS = 48
BASE_CHANNELS = 40
MASK_FRACTION = 0.28
MAX_EPOCHS = 180
EVAL_EVERY = 10
OUTER_SEEDS = (288, 1288, 2288, 3288, 4288)
OUTER_COUNT = 320
INNER_COUNT = 240
FINAL_SEEDS = (28801, 28831, 28861)
MIN_OFFICIAL_DELTA = 5e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grid_coordinates(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.clip(np.rint((pos[:, 0] - X_MIN) / RESOLUTION), 0, GRID_W - 1)
    gy = np.clip(np.rint((pos[:, 1] - Y_MIN) / RESOLUTION), 0, GRID_H - 1)
    return gx.astype(np.int64), gy.astype(np.int64)


def sampling_grid(pos: np.ndarray, device: torch.device) -> torch.Tensor:
    x = 2.0 * (pos[:, 0] - X_MIN) / (X_MAX - X_MIN) - 1.0
    y = 2.0 * (pos[:, 1] - Y_MIN) / (Y_MAX - Y_MIN) - 1.0
    grid = np.stack([x, y], axis=1).astype(np.float32)
    return torch.from_numpy(grid)[None, :, None, :].to(device)


def static_geometry() -> np.ndarray:
    scene = np.load(SCENE_CACHE)
    x = X_MIN + RESOLUTION * np.arange(GRID_W, dtype=np.float32)
    y = Y_MIN + RESOLUTION * np.arange(GRID_H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ix = np.clip(
        np.rint((xx - float(scene["x_min"])) / float(scene["resolution"])),
        0,
        int(scene["grid_nx"]) - 1,
    ).astype(np.int64)
    iy = np.clip(
        np.rint((yy - float(scene["y_min"])) / float(scene["resolution"])),
        0,
        int(scene["grid_ny"]) - 1,
    ).astype(np.int64)

    count = np.log1p(scene["grid_count"][iy, ix].astype(np.float32))
    vertical = np.log1p(
        scene["grid_vertical_count"][iy, ix].astype(np.float32)
    )
    horizontal = np.log1p(
        scene["grid_horizontal_count"][iy, ix].astype(np.float32)
    )
    count /= max(float(np.quantile(count, 0.995)), 1e-6)
    vertical /= max(float(np.quantile(vertical, 0.995)), 1e-6)
    horizontal /= max(float(np.quantile(horizontal, 0.995)), 1e-6)
    z_max = np.nan_to_num(scene["grid_z_max"][iy, ix], nan=0.0).astype(np.float32) / 50.0
    wall_x = scene["grid_wall_nx"][iy, ix].astype(np.float32)
    wall_y = scene["grid_wall_ny"][iy, ix].astype(np.float32)
    component = (scene["component_labels"][iy, ix] > 0).astype(np.float32)
    x_norm = 2.0 * (xx - X_MIN) / (X_MAX - X_MIN) - 1.0
    y_norm = 2.0 * (yy - Y_MIN) / (Y_MAX - Y_MIN) - 1.0
    dx = xx - 50.0
    dy = yy
    radius = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    channels = np.stack(
        [
            np.clip(count, 0.0, 1.5),
            np.clip(vertical, 0.0, 1.5),
            np.clip(horizontal, 0.0, 1.5),
            np.clip(z_max, 0.0, 1.2),
            wall_x,
            wall_y,
            component,
            x_norm,
            y_norm,
            radius / 320.0,
            np.sin(angle),
            np.cos(angle),
        ],
        axis=0,
    ).astype(np.float32)
    return channels


@dataclass
class SpectralBasis:
    mean: np.ndarray
    components: np.ndarray
    coefficient_mean: np.ndarray
    coefficient_std: np.ndarray

    def transform(self, sh: np.ndarray) -> np.ndarray:
        root = np.sqrt(np.maximum(sh.reshape(len(sh), -1), 0.0))
        coefficient = (root - self.mean) @ self.components.T
        return (
            (coefficient - self.coefficient_mean) / self.coefficient_std
        ).astype(np.float32)

    def reconstruct(self, coefficient: np.ndarray) -> np.ndarray:
        raw = (
            coefficient * self.coefficient_std + self.coefficient_mean
        ) @ self.components + self.mean
        return v279.normalize(
            np.maximum(raw, 0.0).reshape(
                len(raw), v279.S_SIZE, v279.H_SIZE
            )
            ** 2
        )


def fit_basis(sh: np.ndarray, fit_ids: np.ndarray, seed: int) -> SpectralBasis:
    root = np.sqrt(
        np.maximum(sh[fit_ids].reshape(len(fit_ids), -1), 0.0)
    ).astype(np.float32)
    pca = PCA(
        n_components=min(PCA_COMPONENTS, len(fit_ids) - 1),
        svd_solver="randomized",
        random_state=seed,
    )
    coefficient = pca.fit_transform(root).astype(np.float32)
    coefficient_mean = coefficient.mean(axis=0).astype(np.float32)
    coefficient_std = coefficient.std(axis=0).astype(np.float32)
    coefficient_std[coefficient_std < 1e-4] = 1.0
    return SpectralBasis(
        pca.mean_.astype(np.float32),
        pca.components_.astype(np.float32),
        coefficient_mean,
        coefficient_std,
    )


def splat_coefficients(
    coefficients: np.ndarray,
    point_ids: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    channels = coefficients.shape[1]
    field = np.zeros((channels, GRID_H, GRID_W), dtype=np.float32)
    density = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    gx, gy = grid_coordinates(positions[point_ids])
    for row, x, y in zip(point_ids, gx, gy):
        field[:, y, x] += coefficients[row]
        density[y, x] += 1.0
    field /= np.maximum(density[None], 1.0)
    density = np.log1p(density)[None]
    return np.concatenate([field, density], axis=0).astype(np.float32)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialFieldUNet(nn.Module):
    def __init__(self, static_channels: int, spectral_channels: int) -> None:
        super().__init__()
        b = BASE_CHANNELS
        self.enc0 = DoubleConv(static_channels + spectral_channels + 1, b)
        self.enc1 = DoubleConv(b, b * 2)
        self.enc2 = DoubleConv(b * 2, b * 3)
        self.enc3 = DoubleConv(b * 3, b * 4)
        self.bottleneck = DoubleConv(b * 4, b * 5)
        self.dec3 = DoubleConv(b * 5 + b * 4, b * 4)
        self.dec2 = DoubleConv(b * 4 + b * 3, b * 3)
        self.dec1 = DoubleConv(b * 3 + b * 2, b * 2)
        self.dec0 = DoubleConv(b * 2 + b, b)
        self.head = nn.Conv2d(b, spectral_channels, 1)

    @staticmethod
    def up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(F.avg_pool2d(e0, 2, ceil_mode=True))
        e2 = self.enc2(F.avg_pool2d(e1, 2, ceil_mode=True))
        e3 = self.enc3(F.avg_pool2d(e2, 2, ceil_mode=True))
        z = self.bottleneck(F.avg_pool2d(e3, 2, ceil_mode=True))
        d3 = self.dec3(torch.cat([self.up(z, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2, e1), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up(d1, e0), e0], dim=1))
        return self.head(d0)


def sample_field(
    field: torch.Tensor, positions: np.ndarray, device: torch.device
) -> torch.Tensor:
    value = F.grid_sample(
        field,
        sampling_grid(positions, device),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return value[0, :, :, 0].T


def reconstruct_torch(
    coefficient: torch.Tensor,
    basis: SpectralBasis,
    device: torch.device,
) -> torch.Tensor:
    std = torch.from_numpy(basis.coefficient_std).to(device)
    cmean = torch.from_numpy(basis.coefficient_mean).to(device)
    components = torch.from_numpy(basis.components).to(device)
    mean = torch.from_numpy(basis.mean).to(device)
    root = (coefficient * std + cmean) @ components + mean
    sh = F.relu(root).square()
    return sh / sh.sum(dim=1, keepdim=True).clamp_min(1e-30)


@torch.no_grad()
def predict_points(
    model: SpatialFieldUNet,
    basis: SpectralBasis,
    coefficients: np.ndarray,
    observed_ids: np.ndarray,
    query_pos: np.ndarray,
    all_pos: np.ndarray,
    static: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dynamic = splat_coefficients(coefficients, observed_ids, all_pos)
    x = torch.cat(
        [static, torch.from_numpy(dynamic)[None].to(device)], dim=1
    )
    field = model(x)
    coefficient = sample_field(field, query_pos, device).cpu().numpy()
    return basis.reconstruct(coefficient), coefficient.astype(np.float32)


def train_model(
    sh: np.ndarray,
    train_pos: np.ndarray,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray | None,
    seed: int,
    epochs: int,
    static: torch.Tensor,
    device: torch.device,
) -> tuple[SpatialFieldUNet, SpectralBasis, dict[str, object]]:
    set_seed(seed)
    basis = fit_basis(sh, fit_ids, seed)
    coefficients = np.zeros((len(sh), len(basis.components)), dtype=np.float32)
    coefficients[:] = basis.transform(sh)
    model = SpatialFieldUNet(static.shape[1], len(basis.components)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        target_count = max(int(round(MASK_FRACTION * len(fit_ids))), 64)
        target_ids = rng.choice(fit_ids, target_count, replace=False)
        target_cells = set(
            zip(*grid_coordinates(train_pos[target_ids]))
        )
        gx, gy = grid_coordinates(train_pos[fit_ids])
        observed_ids = np.asarray(
            [
                point_id
                for point_id, x, y in zip(fit_ids, gx, gy)
                if (int(x), int(y)) not in target_cells
            ],
            dtype=np.int64,
        )
        dynamic = splat_coefficients(coefficients, observed_ids, train_pos)
        x = torch.cat(
            [static, torch.from_numpy(dynamic)[None].to(device)], dim=1
        )
        field = model(x)
        pred_coeff = sample_field(field, train_pos[target_ids], device)
        pred_sh = reconstruct_torch(pred_coeff, basis, device)
        target_sh = torch.from_numpy(
            sh[target_ids].reshape(len(target_ids), -1)
        ).to(device)
        target_d = torch.from_numpy(v279.derive_d_numpy(sh[target_ids])).to(device)
        pred_d = v279.derive_d_torch(pred_sh)
        spectral_loss = (
            0.52
            * (1.0 - F.cosine_similarity(pred_sh, target_sh, dim=1).mean())
            + 0.48
            * (1.0 - F.cosine_similarity(pred_d, target_d, dim=1).mean())
        )
        coefficient_target = torch.from_numpy(coefficients[target_ids]).to(device)
        coefficient_loss = F.smooth_l1_loss(
            pred_coeff, coefficient_target, beta=0.5
        )
        loss = spectral_loss + 0.025 * coefficient_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs:
            item: dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "masked_count": int(len(target_ids)),
                "observed_count": int(len(observed_ids)),
            }
            if validation_ids is not None:
                pred, _ = predict_points(
                    model,
                    basis,
                    coefficients,
                    fit_ids,
                    train_pos[validation_ids],
                    train_pos,
                    static,
                    device,
                )
                metrics = v279.metrics(
                    pred,
                    sh[validation_ids],
                    v279.derive_d_numpy(sh[validation_ids]),
                )
                item["validation"] = metrics
                if metrics["weighted"] > best_metric:
                    best_metric = metrics["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"v288 seed={seed} epoch={epoch}/{epochs} loss={float(loss):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    elif validation_ids is None:
        best_epoch = epochs
    return model, basis, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": (
            float(best_metric) if np.isfinite(best_metric) else None
        ),
        "history": history,
    }


def main() -> None:
    for path in (v283.TARGET_CACHE, SCENE_CACHE, BASE):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_data = np.load(v283.TARGET_CACHE)
    sh = v279.normalize(target_data["sh"].astype(np.float32))
    truth_d = v279.normalize(target_data["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static_np = static_geometry()
    static = torch.from_numpy(static_np)[None].to(device)
    print(
        f"v288 device={device} grid={GRID_H}x{GRID_W} static={static.shape[1]}",
        flush=True,
    )

    outer_rows: list[dict[str, object]] = []
    oof_ids: list[np.ndarray] = []
    oof_base: list[np.ndarray] = []
    oof_model: list[np.ndarray] = []
    selected_epochs: list[int] = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], INNER_COUNT, seed + 50_000
            ),
            outer_fit,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_model, _inner_basis, inner_training = train_model(
            sh,
            train_pos,
            inner_fit,
            inner_ids,
            seed,
            MAX_EPOCHS,
            static,
            device,
        )
        selected_epoch = max(int(inner_training["best_epoch"]), 1)
        selected_epochs.append(selected_epoch)
        del inner_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model, basis, outer_training = train_model(
            sh,
            train_pos,
            outer_fit,
            None,
            seed + 100_000,
            selected_epoch,
            static,
            device,
        )
        coefficients = np.zeros((len(sh), len(basis.components)), dtype=np.float32)
        coefficients[:] = basis.transform(sh)
        prediction, _ = predict_points(
            model,
            basis,
            coefficients,
            outer_fit,
            train_pos[outer_ids],
            train_pos,
            static,
            device,
        )
        baseline_sh = v283.knn_anchor(
            outer_ids, outer_fit, train_pos, sh
        )
        baseline = v279.metrics(
            baseline_sh, sh[outer_ids], truth_d[outer_ids]
        )
        model_metric = v279.metrics(
            prediction, sh[outer_ids], truth_d[outer_ids]
        )
        raw_delta = {
            key: model_metric[key] - baseline[key]
            for key in ("sh", "d", "weighted")
        }
        oracle = v279.metrics(
            basis.reconstruct(basis.transform(sh[outer_ids])),
            sh[outer_ids],
            truth_d[outer_ids],
        )
        outer_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "selected_epoch": selected_epoch,
                "inner_training": inner_training,
                "outer_training": outer_training,
                "pca_oracle": oracle,
                "baseline": baseline,
                "model": model_metric,
                "raw_delta": raw_delta,
            }
        )
        oof_ids.append(outer_ids)
        oof_base.append(baseline_sh)
        oof_model.append(prediction)
        print(
            f"v288 outer={fold} epoch={selected_epoch} "
            f"base={baseline['weighted']:.6f} model={model_metric['weighted']:.6f} "
            f"delta={raw_delta['weighted']:+.6f}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv_ids = np.concatenate(oof_ids)
    cv_base = np.concatenate(oof_base)
    cv_model = np.concatenate(oof_model)
    alpha, alpha_grid = v279.choose_joint_alpha(
        cv_base, cv_model, sh[cv_ids], truth_d[cv_ids]
    )
    fold_deltas: list[dict[str, float]] = []
    for row, ids, base, model in zip(
        outer_rows, oof_ids, oof_base, oof_model
    ):
        blended = v279.normalize(base + alpha * (model - base))
        metric = v279.metrics(blended, sh[ids], truth_d[ids])
        delta = {
            key: metric[key] - row["baseline"][key]
            for key in ("sh", "d", "weighted")
        }
        row["blended"] = metric
        row["blended_delta"] = delta
        fold_deltas.append(delta)
    mean_delta = {
        key: float(np.mean([value[key] for value in fold_deltas]))
        for key in ("sh", "d", "weighted")
    }
    weighted_wins = int(
        np.sum([value["weighted"] > 0.0 for value in fold_deltas])
    )
    official_selected = bool(
        alpha > 0.0
        and mean_delta["weighted"] >= MIN_OFFICIAL_DELTA
        and mean_delta["sh"] > 0.0
        and mean_delta["d"] > 0.0
        and weighted_wins >= 3
    )

    final_training: list[dict[str, object]] = []
    projection_diagnostics = None
    local_score = None
    if official_selected:
        final_epoch = max(int(np.median(selected_epochs)), 1)
        test_predictions: list[np.ndarray] = []
        for seed in FINAL_SEEDS:
            model, basis, training = train_model(
                sh,
                train_pos,
                valid_ids,
                None,
                seed,
                final_epoch,
                static,
                device,
            )
            coefficients = np.zeros(
                (len(sh), len(basis.components)), dtype=np.float32
            )
            coefficients[:] = basis.transform(sh)
            prediction, _ = predict_points(
                model,
                basis,
                coefficients,
                valid_ids,
                test_pos,
                train_pos,
                static,
                device,
            )
            test_predictions.append(prediction)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "model_state": model.state_dict(),
                    "basis": {
                        "mean": basis.mean,
                        "components": basis.components,
                        "coefficient_mean": basis.coefficient_mean,
                        "coefficient_std": basis.coefficient_std,
                    },
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
            del model
        model_sh = v279.normalize(np.mean(test_predictions, axis=0))
        # knn_anchor expects shared query/bank indexing; build the test anchor directly.
        from scipy.spatial import cKDTree

        distance, local = cKDTree(train_pos[valid_ids, :2]).query(
            test_pos[:, :2], k=16
        )
        weight = 1.0 / np.maximum(distance, 0.05) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        test_anchor = v279.normalize(
            np.sum(
                weight[:, :, None, None] * sh[valid_ids[local]],
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)
        )
        current_sh, _ = v274.extract_current_marginals(BASE)
        target_sh = v279.normalize(
            current_sh + alpha * (model_sh - test_anchor)
        )
        target_d = v279.derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            model_sh=model_sh,
            anchor_sh=test_anchor,
            current_sh=current_sh,
            target_sh=target_sh,
            target_d=target_d,
            alpha=np.asarray(alpha, dtype=np.float32),
        )
        projection_diagnostics = v247.project(target_sh, target_d, OUTPUT)
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": (
            "local_gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "A whole-scene planar masked U-Net can inpaint the spectral field across official "
            "wall barriers by jointly observing the point-cloud raster and all available train "
            "labels, unlike independent per-point path or patch predictors."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP labels",
                "official train/test planar coordinates",
                "official point-cloud occupancy/height/normal raster",
                "strict v239 official-only carrier after official selection",
            ],
            "does_not_use": [
                "synthetic/Sionna channel labels",
                "v212/v217 teacher",
                "hidden labels or RT H/path coefficients",
                "local replay truth for training, epoch, or alpha selection",
            ],
            "local_truth_usage": (
                "one final gate score after official selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "grid": [GRID_H, GRID_W],
            "bounds": [X_MIN, X_MAX, Y_MIN, Y_MAX],
            "resolution": RESOLUTION,
            "static_channels": int(static_np.shape[0]),
            "pca_components": PCA_COMPONENTS,
            "mask_fraction": MASK_FRACTION,
            "maximum_epochs": MAX_EPOCHS,
            "outer_seeds": list(OUTER_SEEDS),
            "final_seeds": list(FINAL_SEEDS),
        },
        "outer_folds": outer_rows,
        "official_selection": {
            "residual_alpha": alpha,
            "alpha_grid": alpha_grid,
            "mean_delta": mean_delta,
            "weighted_wins": weighted_wins,
            "selected_epochs": selected_epochs,
            "selected": official_selected,
        },
        "final_training": final_training,
        "projection_diagnostics": (
            {
                key: summarize(value)
                for key, value in projection_diagnostics.items()
            }
            if projection_diagnostics is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(
                    float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
                ),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "targets": (
                PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None
            ),
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
