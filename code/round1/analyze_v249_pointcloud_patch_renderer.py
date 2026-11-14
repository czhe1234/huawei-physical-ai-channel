#!/usr/bin/env python3
"""v249: raw official point-cloud multiscale patch path renderer.

Each UE receives three BS-aligned occupancy/height/normal patches sampled from
the official point-cloud grid.  A CNN embeds these raw scene tensors and feeds
the embedding into the differentiable path renderer from v248.  Training and
selection use only official H and official geometry.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v248_differentiable_path_renderer as v248
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 249
VERSION = "v249_pointcloud_patch_renderer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v249_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v249_pointcloud_patch_targets.npz"
ACTIONS = OUT_DIR / "v249_pointcloud_patch_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
TRAIN_PATCHES = OUT_DIR / "v249_train_pointcloud_patches.npy"
TEST_PATCHES = OUT_DIR / "v249_test_pointcloud_patches.npy"

SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

LOCAL_GATE = 0.645604
BS_XY = np.asarray([50.0, 0.0], dtype=np.float32)
PATCH_SIZE = 32
PATCH_EXTENTS = (20.0, 60.0, 160.0)
PATCH_CHANNELS_PER_SCALE = 7
VAL_COUNT = 320
TRAIN_EPOCHS = 240
BATCH_SIZE = 48
FINAL_SEEDS = (24901, 24923, 24953)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_grid_channels(scene: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    count = np.log1p(scene["grid_count"].astype(np.float32))
    vertical = np.log1p(scene["grid_vertical_count"].astype(np.float32))
    horizontal = np.log1p(scene["grid_horizontal_count"].astype(np.float32))
    z_max = np.nan_to_num(scene["grid_z_max"].astype(np.float32), nan=0.0)
    count /= max(float(np.quantile(count, 0.995)), 1e-6)
    vertical /= max(float(np.quantile(vertical, 0.995)), 1e-6)
    horizontal /= max(float(np.quantile(horizontal, 0.995)), 1e-6)
    z_max = np.clip(z_max / 45.0, 0.0, 1.5)
    component = (scene["component_labels"].astype(np.int32) > 0).astype(np.float32)
    return {
        "count": np.clip(count, 0.0, 1.5),
        "vertical": np.clip(vertical, 0.0, 1.5),
        "horizontal": np.clip(horizontal, 0.0, 1.5),
        "z_max": z_max,
        "wall_nx": scene["grid_wall_nx"].astype(np.float32),
        "wall_ny": scene["grid_wall_ny"].astype(np.float32),
        "component": component,
    }


def build_patches(pos: np.ndarray, output_path: Path) -> np.ndarray:
    if output_path.is_file():
        return np.load(output_path, mmap_mode="r")
    scene = np.load(SCENE_CACHE)
    channels = normalized_grid_channels(scene)
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(pos), len(PATCH_EXTENTS) * PATCH_CHANNELS_PER_SCALE, PATCH_SIZE, PATCH_SIZE),
    )
    axis = np.linspace(-1.0, 1.0, PATCH_SIZE, dtype=np.float32)
    uu, vv = np.meshgrid(axis, axis, indexing="ij")
    scalar_names = ("count", "vertical", "horizontal", "z_max", "component")
    for index, point in enumerate(pos):
        direction = BS_XY - point[:2]
        norm = max(float(np.linalg.norm(direction)), 1e-6)
        direction = direction / norm
        tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        patch_parts = []
        for extent in PATCH_EXTENTS:
            world_x = point[0] + extent * (uu * direction[0] + vv * tangent[0])
            world_y = point[1] + extent * (uu * direction[1] + vv * tangent[1])
            grid_x = (world_x - x_min) / resolution
            grid_y = (world_y - y_min) / resolution
            coords = np.stack([grid_y, grid_x], axis=0)
            sampled = [
                ndimage.map_coordinates(channels[name], coords, order=1, mode="constant", cval=0.0)
                for name in scalar_names
            ]
            nx = ndimage.map_coordinates(
                channels["wall_nx"], coords, order=1, mode="constant", cval=0.0
            )
            ny = ndimage.map_coordinates(
                channels["wall_ny"], coords, order=1, mode="constant", cval=0.0
            )
            normal_parallel = nx * direction[0] + ny * direction[1]
            normal_perpendicular = nx * tangent[0] + ny * tangent[1]
            patch_parts.extend([*sampled, normal_parallel, normal_perpendicular])
        output[index] = np.stack(patch_parts, axis=0).astype(np.float16)
        if (index + 1) % 100 == 0 or index + 1 == len(pos):
            output.flush()
            print(f"v249 pointcloud patches {output_path.name} {index + 1}/{len(pos)}", flush=True)
    return np.load(output_path, mmap_mode="r")


@dataclass
class PatchRendererData:
    base: v248.RendererData
    patches: np.ndarray


class PointCloudPatchRenderer(v248.DifferentiablePathRenderer):
    def __init__(self, global_dim: int, path_dim: int, patch_channels: int):
        patch_dim = 128
        super().__init__(global_dim + patch_dim, path_dim)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(patch_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.GroupNorm(12, 96),
            nn.Conv2d(96, 144, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.GroupNorm(12, 144),
            nn.Conv2d(144, 192, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.GroupNorm(16, 192),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, patch_dim),
            nn.LayerNorm(patch_dim),
        )

    def forward_with_patches(
        self,
        global_features: torch.Tensor,
        patches: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        delay: torch.Tensor,
        tx_dir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        patch_features = self.patch_encoder(patches)
        return super().forward(
            torch.cat([global_features, patch_features], dim=1),
            path_numeric,
            path_type,
            path_valid,
            delay,
            tx_dir,
        )


@torch.no_grad()
def predict(
    model: PointCloudPatchRenderer,
    data: PatchRendererData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    sh_parts = []
    d_parts = []
    diagnostic_parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(data.base.global_features), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(data.base.global_features))
        sh, d, diagnostics = model.forward_with_patches(
            torch.from_numpy(data.base.global_features[start:end]).to(device),
            torch.from_numpy(np.asarray(data.patches[start:end], dtype=np.float32)).to(device),
            torch.from_numpy(data.base.path_numeric[start:end]).to(device),
            torch.from_numpy(data.base.path_type[start:end]).to(device),
            torch.from_numpy(data.base.path_valid[start:end]).to(device),
            torch.from_numpy(data.base.delay[start:end]).to(device),
            torch.from_numpy(data.base.tx_dir[start:end]).to(device),
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
    data: PatchRendererData,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: PatchRendererData | None,
) -> tuple[PointCloudPatchRenderer, dict[str, object]]:
    set_seed(seed)
    model = PointCloudPatchRenderer(
        data.base.global_features.shape[1],
        data.base.path_numeric.shape[2],
        data.patches.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=2.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    n = len(data.base.global_features)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            pred_sh, pred_d, diagnostics = model.forward_with_patches(
                torch.from_numpy(data.base.global_features[ids]).to(device),
                torch.from_numpy(np.asarray(data.patches[ids], dtype=np.float32)).to(device),
                torch.from_numpy(data.base.path_numeric[ids]).to(device),
                torch.from_numpy(data.base.path_type[ids]).to(device),
                torch.from_numpy(data.base.path_valid[ids]).to(device),
                torch.from_numpy(data.base.delay[ids]).to(device),
                torch.from_numpy(data.base.tx_dir[ids]).to(device),
            )
            loss = v248.loss_function(
                pred_sh,
                pred_d,
                torch.from_numpy(data.base.target_sh[ids]).to(device),
                torch.from_numpy(data.base.target_d[ids]).to(device),
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
                val_metrics = v248.spectral_metrics(
                    val_sh,
                    val_d,
                    validation.base.target_sh,
                    validation.base.target_d,
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
            print(f"v249 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": None if not np.isfinite(best_metric) else float(best_metric),
        "history": history,
    }


def main() -> None:
    for path in [SCENE_CACHE, TARGET_CACHE, GEOM_CACHE, v248.PATH_CACHE, v247.BASE]:
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
    train_patches = build_patches(train_pos, TRAIN_PATCHES)
    test_patches = build_patches(test_pos, TEST_PATCHES)

    matched = choose_val_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, "matched-test", VERSION_NUMBER
    )
    val_ids = np.intersect1d(matched, valid_ids)
    fit_ids = np.setdiff1d(valid_ids, val_ids)
    fit_raw = v248.make_raw_data(
        "train", fit_ids, train_pos[fit_ids], train_blocked[fit_ids], sh[fit_ids], d[fit_ids]
    )
    val_raw = v248.make_raw_data(
        "train", val_ids, train_pos[val_ids], train_blocked[val_ids], sh[val_ids], d[val_ids]
    )
    stats = v248.fit_feature_stats(fit_raw)
    fit_data = PatchRendererData(v248.apply_feature_stats(fit_raw, stats), train_patches[fit_ids])
    val_data = PatchRendererData(v248.apply_feature_stats(val_raw, stats), train_patches[val_ids])
    validation_model, validation_training = train_model(
        fit_data, FINAL_SEEDS[0], TRAIN_EPOCHS, device, val_data
    )
    val_sh, val_d, val_diag = predict(validation_model, val_data, device)
    validation_metrics = v248.spectral_metrics(
        val_sh, val_d, val_data.base.target_sh, val_data.base.target_d
    )
    selected_epochs = max(20, int(validation_training["best_epoch"]))
    del validation_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_raw = v248.make_raw_data(
        "train",
        valid_ids,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        sh[valid_ids],
        d[valid_ids],
    )
    test_raw = v248.make_raw_data(
        "test",
        None,
        test_pos,
        test_blocked,
        np.zeros((len(test_pos), v248.S_SC, v248.MH), dtype=np.float32),
        np.zeros((len(test_pos), v248.S_SC), dtype=np.float32),
    )
    final_stats = v248.fit_feature_stats(final_raw)
    final_data = PatchRendererData(
        v248.apply_feature_stats(final_raw, final_stats), train_patches[valid_ids]
    )
    test_data = PatchRendererData(v248.apply_feature_stats(test_raw, final_stats), test_patches)
    seed_sh = []
    seed_d = []
    seed_diag = []
    final_training = []
    for seed in FINAL_SEEDS:
        model, info = train_model(final_data, seed, selected_epochs, device, None)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "epochs": selected_epochs,
                "feature_stats": final_stats,
                "patch_extents": PATCH_EXTENTS,
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
            "Raw BS-aligned multiscale occupancy/height/normal patches contain wall, corridor, "
            "and edge context lost by the 15-path summary, allowing the official path renderer "
            "to infer propagation mode and spectral weights more accurately."
        ),
        "compliance_boundary": {
            "generation_uses": [
                "official train H leaked S-H and PDP targets",
                "official train/test coordinates",
                "raw official point-cloud-derived occupancy/height/normal grids",
                "official point-cloud path candidates",
                "v239 strict official-data carrier",
            ],
            "generation_does_not_use": [
                "Sionna or synthetic channel labels",
                "v212/v217/v221/v222 teacher outputs",
                "local replay truth during training or selection",
                "hidden labels",
                "raw RT H",
            ],
            "local_truth_usage": "one final diagnostic score after generation",
        },
        "parameters": {
            "patch_size": PATCH_SIZE,
            "patch_extents": list(PATCH_EXTENTS),
            "patch_channels": int(train_patches.shape[1]),
            "valid_train_count": int(len(valid_ids)),
            "fit_count": int(len(fit_ids)),
            "validation_count": int(len(val_ids)),
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
            "train_patches": PATHS.relative(TRAIN_PATCHES),
            "test_patches": PATHS.relative(TEST_PATCHES),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
