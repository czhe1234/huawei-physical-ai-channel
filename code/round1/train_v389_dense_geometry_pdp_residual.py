#!/usr/bin/env python3
"""Train a dense-city geometry residual directly on normalized global PDP D."""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

import build_v382_dense_lifecycle_examples as v382
import build_v383_isolation_matched_lifecycle_examples as v383
import lifecycle_geometry_features as lgf
from project_paths import PATHS


VERSION_NUMBER = 389
VERSION = "v389_dense_geometry_pdp_residual"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
DATASET_CACHE = OUT_DIR / "v389_dense_geometry_pdp_dataset.npz"
CHECKPOINT = OUT_DIR / "v389_dense_geometry_pdp_residual_model.pt"
SCENE_ROOT = PATHS.cache_dir(381) / "scenes"

N_DELAY = 192
NEIGHBORS = 16
DISTANCE_SCALE = 20.0
MAX_EPOCHS = 50
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-5
RESIDUAL_SCALE = 1.5

MIN_COSINE_LIFT = 0.01
MIN_FULL_VS_SHUFFLED = 0.005
MIN_EMD_REDUCTION = 0.05
MIN_SCENE_WINS = 4


def normalize_rows(value: np.ndarray) -> np.ndarray:
    return np.divide(
        value,
        np.maximum(np.sum(value, axis=1, keepdims=True), 1e-20),
        out=np.zeros_like(value),
        where=np.sum(value, axis=1, keepdims=True) > 0.0,
    )


def dirichlet_power(delay: float) -> np.ndarray:
    bins = np.arange(N_DELAY, dtype=np.float64)
    delta = bins - float(delay)
    denominator = np.sin(np.pi * delta / N_DELAY)
    numerator = np.sin(np.pi * delta)
    kernel = np.empty(N_DELAY, dtype=np.float64)
    close = np.abs(denominator) < 1e-10
    kernel[close] = 1.0
    kernel[~close] = (numerator[~close] / (N_DELAY * denominator[~close])) ** 2
    kernel = np.maximum(kernel, 0.0)
    return kernel / max(float(np.sum(kernel)), 1e-30)


def render_scene_d(data: dict[str, np.ndarray]) -> np.ndarray:
    valid = data["valid"].astype(bool)
    delay = data["delay_bin"].astype(np.float64)
    mass = data["normalized_power"].astype(np.float64)
    output = np.zeros((valid.shape[0], N_DELAY), dtype=np.float64)
    for point_id in range(valid.shape[0]):
        for family_id in np.flatnonzero(valid[point_id] & (mass[point_id] > 0.0)):
            output[point_id] += mass[point_id, family_id] * dirichlet_power(
                delay[point_id, family_id] % N_DELAY
            )
    return normalize_rows(output).astype(np.float32)


def knn_baseline(
    position: np.ndarray,
    truth: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = np.zeros_like(truth)
    variation = np.zeros_like(truth)
    context = np.zeros((len(position), 12), dtype=np.float32)
    ids = np.arange(len(position), dtype=np.int64)
    for query_ids in folds:
        fit_ids = np.setdiff1d(ids, query_ids, assume_unique=True)
        take = min(NEIGHBORS, len(fit_ids))
        distance, local = cKDTree(position[fit_ids, :2]).query(
            position[query_ids, :2], k=take
        )
        if take == 1:
            distance = distance[:, None]
            local = local[:, None]
        donor = truth[fit_ids[local]]
        weight = np.exp(-distance / DISTANCE_SCALE)
        weight /= np.maximum(np.sum(weight, axis=1, keepdims=True), 1e-20)
        mean = np.sum(weight[:, :, None] * donor, axis=1)
        variance = np.sum(
            weight[:, :, None] * (donor - mean[:, None, :]) ** 2, axis=1
        )
        baseline[query_ids] = normalize_rows(mean.astype(np.float32))
        variation[query_ids] = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        entropy = -np.sum(
            baseline[query_ids] * np.log(np.maximum(baseline[query_ids], 1e-8)),
            axis=1,
        ) / np.log(N_DELAY)
        peak = np.max(baseline[query_ids], axis=1)
        variation_l1 = np.sum(variation[query_ids], axis=1)
        context[query_ids] = np.column_stack(
            [
                distance[:, 0] / 100.0,
                np.mean(distance[:, : min(4, take)], axis=1) / 100.0,
                np.mean(distance[:, : min(8, take)], axis=1) / 100.0,
                np.mean(distance, axis=1) / 100.0,
                np.max(distance, axis=1) / 100.0,
                np.std(distance, axis=1) / 100.0,
                entropy,
                peak,
                variation_l1,
                np.max(variation[query_ids], axis=1),
                np.mean(weight[:, : min(4, take)], axis=1),
                np.max(weight, axis=1),
            ]
        ).astype(np.float32)
    return baseline, variation, context


def build_dataset() -> dict[str, np.ndarray]:
    if DATASET_CACHE.is_file():
        cached = np.load(DATASET_CACHE)
        return {key: cached[key] for key in cached.files}
    rows: dict[str, list[np.ndarray]] = {
        "truth": [],
        "baseline": [],
        "variation": [],
        "geometry": [],
        "context": [],
        "scene": [],
        "point": [],
    }
    for scene_id in range(30):
        path = SCENE_ROOT / f"scene_{scene_id:03d}" / "lifecycle.npz"
        data_cache = np.load(path)
        data = {key: data_cache[key] for key in data_cache.files}
        position = data["positions"].astype(np.float64)
        truth = render_scene_d(data)
        folds = v383.isolation_folds(position, 389_000 + scene_id)
        baseline, variation, context = knn_baseline(position, truth, folds)
        raster = lgf.raster_from_boxes(data["boxes"].astype(np.float64))
        geometry = lgf.point_features(
            raster, position, data["bs"].astype(np.float64)
        )
        rows["truth"].append(truth)
        rows["baseline"].append(baseline)
        rows["variation"].append(variation)
        rows["geometry"].append(geometry.astype(np.float32))
        rows["context"].append(context)
        rows["scene"].append(np.full(len(position), scene_id, dtype=np.int8))
        rows["point"].append(np.arange(len(position), dtype=np.int16))
        cosine = np.sum(truth * baseline, axis=1) / np.maximum(
            np.linalg.norm(truth, axis=1) * np.linalg.norm(baseline, axis=1), 1e-20
        )
        print(
            f"v389 dataset scene={scene_id} baseline_cos={np.mean(cosine):.4f}",
            flush=True,
        )
    output = {key: np.concatenate(value) for key, value in rows.items()}
    np.savez_compressed(DATASET_CACHE, **output)
    return output


class SpectrumResidualNet(nn.Module):
    def __init__(self, geometry_dim: int):
        super().__init__()
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.spectrum = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(64, 64, 5, padding=2, padding_mode="circular"),
            nn.SiLU(),
        )
        self.film = nn.Linear(128, 128)
        self.head = nn.Sequential(
            nn.Conv1d(64, 64, 5, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(64, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        spectrum: torch.Tensor,
        geometry: torch.Tensor,
        baseline_log: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.spectrum(spectrum)
        film = self.film(self.geometry(geometry))
        scale, shift = torch.chunk(film, 2, dim=1)
        hidden = hidden * (1.0 + 0.25 * torch.tanh(scale)[:, :, None])
        hidden = hidden + 0.25 * shift[:, :, None]
        residual = RESIDUAL_SCALE * torch.tanh(self.head(hidden).squeeze(1))
        return torch.softmax(baseline_log + residual, dim=1)


def spectral_input(baseline: np.ndarray, variation: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.log(np.maximum(baseline, 1e-8)) / 8.0,
            np.log1p(100.0 * variation) / 4.0,
        ],
        axis=1,
    ).astype(np.float32)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    cosine = np.sum(truth * prediction, axis=1) / np.maximum(
        np.linalg.norm(truth, axis=1) * np.linalg.norm(prediction, axis=1), 1e-20
    )
    emd = np.mean(np.abs(np.cumsum(truth, axis=1) - np.cumsum(prediction, axis=1)))
    return {
        "cosine": float(np.mean(cosine)),
        "emd": float(emd),
    }


def predict(
    model: SpectrumResidualNet,
    spectrum: np.ndarray,
    geometry: np.ndarray,
    baseline: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    output = np.empty_like(baseline)
    baseline_log = np.log(np.maximum(baseline, 1e-8)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(baseline), 4096):
            end = min(start + 4096, len(baseline))
            output[start:end] = model(
                torch.from_numpy(spectrum[start:end]).to(device),
                torch.from_numpy(geometry[start:end]).to(device),
                torch.from_numpy(baseline_log[start:end]).to(device),
            ).cpu().numpy()
    return output


def train_model(
    *,
    name: str,
    spectrum: np.ndarray,
    geometry: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    train_keep: np.ndarray,
    validation_keep: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[SpectrumResidualNet, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpectrumResidualNet(geometry.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_rows = np.flatnonzero(train_keep)
    validation_rows = np.flatnonzero(validation_keep)
    baseline_log = np.log(np.maximum(baseline, 1e-8)).astype(np.float32)
    rng = np.random.default_rng(seed)
    best_state = None
    best_cosine = -np.inf
    selected_epoch = 0
    epoch_rows = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            spectrum_batch = torch.from_numpy(spectrum[rows]).to(device)
            geometry_batch = torch.from_numpy(geometry[rows]).to(device)
            baseline_log_batch = torch.from_numpy(baseline_log[rows]).to(device)
            truth_batch = torch.from_numpy(truth[rows]).to(device)
            prediction = model(
                spectrum_batch, geometry_batch, baseline_log_batch
            )
            cosine = nn.functional.cosine_similarity(prediction, truth_batch, dim=1)
            kl = torch.sum(
                truth_batch
                * (
                    torch.log(torch.clamp(truth_batch, min=1e-8))
                    - torch.log(torch.clamp(prediction, min=1e-8))
                ),
                dim=1,
            )
            cdf = torch.mean(
                (
                    torch.cumsum(prediction, dim=1)
                    - torch.cumsum(truth_batch, dim=1)
                )
                ** 2,
                dim=1,
            )
            loss = torch.mean(1.0 - cosine + 0.10 * kl + 2.0 * cdf)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = predict(
            model,
            spectrum[validation_rows],
            geometry[validation_rows],
            baseline[validation_rows],
            device,
        )
        validation = metrics(truth[validation_rows], validation_prediction)
        epoch_rows.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), **validation}
        )
        if validation["cosine"] > best_cosine:
            best_cosine = validation["cosine"]
            selected_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"v389 {name} epoch={epoch} loss={np.mean(losses):.4f} "
                f"val_cos={validation['cosine']:.5f} emd={validation['emd']:.5f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("v389 selected no model")
    model.load_state_dict(best_state)
    return model, {
        "selected_epoch": selected_epoch,
        "best_validation_cosine": float(best_cosine),
        "epoch_rows": epoch_rows,
    }


def main() -> None:
    data = build_dataset()
    truth = data["truth"].astype(np.float32)
    baseline = data["baseline"].astype(np.float32)
    variation = data["variation"].astype(np.float32)
    scene = data["scene"].astype(np.int64)
    geometry_raw = np.concatenate(
        [data["geometry"], data["context"]], axis=1
    ).astype(np.float32)
    train_keep = np.isin(scene, v382.TRAIN_SCENES)
    validation_keep = np.isin(scene, v382.VALIDATION_SCENES)
    test_keep = np.isin(scene, v382.TEST_SCENES)
    mean = np.mean(geometry_raw[train_keep], axis=0).astype(np.float32)
    std = np.maximum(
        np.std(geometry_raw[train_keep], axis=0).astype(np.float32), 1e-4
    )
    geometry = np.clip((geometry_raw - mean) / std, -8.0, 8.0).astype(np.float32)
    shuffled_geometry = geometry.copy()
    rng = np.random.default_rng(38901)
    point_columns = lgf.feature_dimensions()["point"]
    for scene_id in np.unique(scene):
        rows = np.flatnonzero(scene == scene_id)
        order = rng.permutation(len(rows))
        shuffled_geometry[np.ix_(rows, np.arange(point_columns))] = geometry[
            np.ix_(rows[order], np.arange(point_columns))
        ]
    spectrum = spectral_input(baseline, variation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model, full_training = train_model(
        name="full",
        spectrum=spectrum,
        geometry=geometry,
        baseline=baseline,
        truth=truth,
        train_keep=train_keep,
        validation_keep=validation_keep,
        device=device,
        seed=38911,
    )
    shuffled_model, shuffled_training = train_model(
        name="shuffled_geometry",
        spectrum=spectrum,
        geometry=shuffled_geometry,
        baseline=baseline,
        truth=truth,
        train_keep=train_keep,
        validation_keep=validation_keep,
        device=device,
        seed=38911,
    )
    test_rows = np.flatnonzero(test_keep)
    full_prediction = predict(
        full_model,
        spectrum[test_rows],
        geometry[test_rows],
        baseline[test_rows],
        device,
    )
    shuffled_prediction = predict(
        shuffled_model,
        spectrum[test_rows],
        shuffled_geometry[test_rows],
        baseline[test_rows],
        device,
    )
    baseline_test = metrics(truth[test_rows], baseline[test_rows])
    full_test = metrics(truth[test_rows], full_prediction)
    shuffled_test = metrics(truth[test_rows], shuffled_prediction)
    scene_rows = []
    for scene_id in v382.TEST_SCENES:
        keep = scene[test_rows] == scene_id
        base = metrics(truth[test_rows][keep], baseline[test_rows][keep])
        full = metrics(truth[test_rows][keep], full_prediction[keep])
        shuffled = metrics(truth[test_rows][keep], shuffled_prediction[keep])
        scene_rows.append(
            {
                "scene": int(scene_id),
                "baseline": base,
                "full": full,
                "shuffled": shuffled,
                "cosine_lift": float(full["cosine"] - base["cosine"]),
                "full_vs_shuffled": float(full["cosine"] - shuffled["cosine"]),
            }
        )
    cosine_lift = full_test["cosine"] - baseline_test["cosine"]
    full_vs_shuffled = full_test["cosine"] - shuffled_test["cosine"]
    emd_reduction = (baseline_test["emd"] - full_test["emd"]) / max(
        baseline_test["emd"], 1e-20
    )
    scene_wins = int(
        sum(
            row["cosine_lift"] > 0.0 and row["full_vs_shuffled"] > 0.0
            for row in scene_rows
        )
    )
    passed = bool(
        cosine_lift >= MIN_COSINE_LIFT
        and full_vs_shuffled >= MIN_FULL_VS_SHUFFLED
        and emd_reduction >= MIN_EMD_REDUCTION
        and scene_wins >= MIN_SCENE_WINS
    )
    torch.save(
        {
            "state_dict": full_model.state_dict(),
            "geometry_mean": mean,
            "geometry_std": std,
            "selected_epoch": full_training["selected_epoch"],
            "parameters": {
                "geometry_dimension": int(geometry.shape[1]),
                "neighbors": NEIGHBORS,
                "distance_scale": DISTANCE_SCALE,
                "residual_scale": RESIDUAL_SCALE,
            },
        },
        CHECKPOINT,
    )
    report = {
        "method": VERSION,
        "status": "synthetic_pdp_residual_passed" if passed else "analysis_only_no_transfer",
        "scope": "unseen dense-scene normalized global-D residual",
        "primary_hypothesis": (
            "Public-geometry features predict the normalized PDP residual of an "
            "official-matched local KNN baseline without explicit path-family transfer."
        ),
        "parameters": {
            "delay_bins": N_DELAY,
            "renderer": "periodic Dirichlet power kernel",
            "neighbors": NEIGHBORS,
            "distance_scale": DISTANCE_SCALE,
            "epochs": MAX_EPOCHS,
            "residual_scale": RESIDUAL_SCALE,
        },
        "training": {
            "full": full_training,
            "shuffled_geometry": shuffled_training,
        },
        "test": {
            "baseline": baseline_test,
            "full": full_test,
            "shuffled_geometry": shuffled_test,
            "cosine_lift": float(cosine_lift),
            "full_vs_shuffled": float(full_vs_shuffled),
            "emd_reduction": float(emd_reduction),
            "scene_rows": scene_rows,
            "scene_wins": scene_wins,
        },
        "decision_rule": {
            "cosine_lift": f">={MIN_COSINE_LIFT}",
            "full_vs_shuffled": f">={MIN_FULL_VS_SHUFFLED}",
            "emd_reduction": f">={MIN_EMD_REDUCTION}",
            "scene_wins": f">={MIN_SCENE_WINS}/4",
            "passed": passed,
        },
        "outputs": {
            "dataset": PATHS.relative(DATASET_CACHE),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["test"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_rule"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
