#!/usr/bin/env python3
"""Learn official array fingerprints from high-confidence one-bounce geometry."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v273_planar_barrier_kernel as v273
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v309_unmatched_effective_scatterers as v309
import analyze_v315_full_sequence_confirmatory as v315
from project_paths import PATHS


VERSION_NUMBER = 318
VERSION = "v318_geometry_array_fingerprint_manifold"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE = OUT_DIR / "v318_onebounce_fingerprint_dataset.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
EPOCHS = 100
BATCH_SIZE = 256


class FingerprintMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Dropout(0.05),
            nn.Linear(192, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 30),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.network(x)
        return (
            torch.softmax(output[:, :16], dim=1),
            torch.softmax(output[:, 16:24], dim=1),
            torch.softmax(output[:, 24:28], dim=1),
            torch.softmax(output[:, 28:30], dim=1),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset() -> dict[str, np.ndarray]:
    if DATA_CACHE.is_file():
        data = np.load(DATA_CACHE)
        return {key: data[key] for key in data.files}
    match = np.load(v305.MATCH_CACHE)
    good = (
        (match["length_error_bins"] <= v309.GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= v309.GOOD_H_ERROR)
    )
    point_id = match["point_id"][good].astype(np.int64)
    component_id = match["component_id"][good].astype(np.int64)
    candidate_id = match["candidate_id"][good].astype(np.int64)
    mass = match["mass"][good].astype(np.float32)
    confidence = np.exp(-0.5 * match["cost"][good]).astype(np.float32)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    candidate_data = np.load(v306.TRAIN_CANDIDATES)
    reflection = candidate_data["reflection_point"][point_id, candidate_id].astype(np.float32)
    path_length = candidate_data["path_length"][point_id, candidate_id].astype(np.float32)
    excess = candidate_data["excess_length"][point_id, candidate_id].astype(np.float32)
    patch_id = candidate_data["patch_id"][point_id, candidate_id].astype(np.int64)
    scene = np.load(v306.SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    patch = patches[np.clip(patch_id, 0, len(patches) - 1)]
    bs_to_reflection = reflection - v304.BS.astype(np.float32)[None, :]
    reflection_to_ue = train_pos[point_id] - reflection
    ue_to_bs = v304.BS.astype(np.float32)[None, :] - train_pos[point_id]
    departure = bs_to_reflection / np.maximum(
        np.linalg.norm(bs_to_reflection, axis=1, keepdims=True), 1e-6
    )
    arrival = reflection_to_ue / np.maximum(
        np.linalg.norm(reflection_to_ue, axis=1, keepdims=True), 1e-6
    )
    ue_boresight = ue_to_bs / np.maximum(
        np.linalg.norm(ue_to_bs, axis=1, keepdims=True), 1e-6
    )
    normal = np.concatenate([patch[:, :2], np.zeros((len(patch), 1), dtype=np.float32)], axis=1)
    incidence = np.abs(np.sum(departure * normal, axis=1))
    outgoing = np.abs(np.sum(arrival * normal, axis=1))
    direct = np.linalg.norm(train_pos[point_id] - v304.BS.astype(np.float32), axis=1)
    numeric = np.concatenate(
        [
            departure,
            arrival,
            ue_boresight,
            (train_pos[point_id] - v304.BS.astype(np.float32)[None, :]) / 500.0,
            (reflection - v304.BS.astype(np.float32)[None, :]) / 500.0,
            np.stack(
                [
                    path_length / 800.0,
                    excess / 400.0,
                    direct / 500.0,
                    incidence,
                    outgoing,
                    patch[:, 0],
                    patch[:, 1],
                    patch[:, 5] / 40.0,
                    patch[:, 6] / 40.0,
                    np.log1p(patch[:, 7]) / 10.0,
                ],
                axis=1,
            ),
        ],
        axis=1,
    ).astype(np.float32)
    fingerprint = np.load(v300.FINGERPRINT_CACHE)
    payload = {
        "point_id": point_id.astype(np.int32),
        "component_id": component_id.astype(np.int16),
        "candidate_id": candidate_id.astype(np.int16),
        "patch_id": patch_id.astype(np.int32),
        "numeric": numeric,
        "horizontal": fingerprint["horizontal"][point_id, component_id].astype(np.float32),
        "vertical": fingerprint["vertical"][point_id, component_id].astype(np.float32),
        "ue_port": fingerprint["ue_port"][point_id, component_id].astype(np.float32),
        "polarization": fingerprint["polarization"][point_id, component_id].astype(np.float32),
        "mass": mass,
        "confidence": confidence,
    }
    np.savez_compressed(DATA_CACHE, **payload)
    return payload


def profile_cosine(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sum(prediction * truth, axis=1) / np.maximum(
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1), 1e-30
    )


def train_model(
    x: np.ndarray,
    targets: tuple[np.ndarray, ...],
    weight: np.ndarray,
    seed: int,
    device: torch.device,
) -> FingerprintMLP:
    set_seed(seed)
    model = FingerprintMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    x_tensor = torch.from_numpy(x).to(device)
    target_tensor = tuple(torch.from_numpy(value).to(device) for value in targets)
    weight_tensor = torch.from_numpy(weight).to(device)
    rng = np.random.default_rng(seed)
    for _epoch in range(EPOCHS):
        order = rng.permutation(len(x))
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            ids = torch.from_numpy(order[start : start + BATCH_SIZE]).to(device)
            prediction = model(x_tensor[ids])
            losses = []
            for current, truth in zip(prediction, target_tensor):
                cosine = 1.0 - F.cosine_similarity(current, truth[ids], dim=1)
                losses.append(cosine)
            per_sample = 0.20 * losses[0] + 0.35 * losses[1] + 0.25 * losses[2] + 0.20 * losses[3]
            loss = torch.sum(per_sample * weight_tensor[ids]) / torch.sum(weight_tensor[ids]).clamp_min(1e-20)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict(model: FingerprintMLP, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, ...]:
    model.eval()
    output = [[] for _ in range(4)]
    for start in range(0, len(x), 2048):
        current = model(torch.from_numpy(x[start : start + 2048]).to(device))
        for index, value in enumerate(current):
            output[index].append(value.cpu().numpy())
    return tuple(np.concatenate(value) for value in output)


def evaluate(prediction: tuple[np.ndarray, ...], truth: tuple[np.ndarray, ...]) -> dict[str, float]:
    names = ("horizontal", "vertical", "ue_port", "polarization")
    return {
        name: float(np.mean(profile_cosine(current, target)))
        for name, current, target in zip(names, prediction, truth)
    }


def main() -> None:
    data = build_dataset()
    target_tuple = (
        data["horizontal"],
        data["vertical"],
        data["ue_port"],
        data["polarization"],
    )
    target_data = np.load(v273.TARGET_CACHE)
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows = []
    pooled_truth = [[] for _ in range(4)]
    pooled_prediction = [[] for _ in range(4)]
    pooled_mean = [[] for _ in range(4)]
    pooled_shuffle = [[] for _ in range(4)]
    for fold, query_ids in enumerate(folds):
        query = np.isin(data["point_id"], query_ids)
        fit = ~query
        mean = tuple(
            np.broadcast_to(
                np.average(value[fit], axis=0, weights=data["confidence"][fit]),
                (int(np.sum(query)), value.shape[1]),
            ).copy().astype(np.float32)
            for value in target_tuple
        )
        x_mean = data["numeric"][fit].mean(axis=0)
        x_std = data["numeric"][fit].std(axis=0)
        x_std[x_std < 1e-5] = 1.0
        fit_x = ((data["numeric"][fit] - x_mean) / x_std).astype(np.float32)
        query_x = ((data["numeric"][query] - x_mean) / x_std).astype(np.float32)
        fit_targets = tuple(value[fit] for value in target_tuple)
        query_targets = tuple(value[query] for value in target_tuple)
        weight = np.sqrt(np.maximum(data["mass"][fit], 1e-5)) * data["confidence"][fit]
        current_model = train_model(fit_x, fit_targets, weight.astype(np.float32), 318 + fold, device)
        current_prediction = predict(current_model, query_x, device)
        rng = np.random.default_rng(31_800 + fold)
        permutation = rng.permutation(np.sum(fit))
        shuffled_targets = tuple(value[permutation] for value in fit_targets)
        shuffled_model = train_model(
            fit_x, shuffled_targets, weight.astype(np.float32), 1318 + fold, device
        )
        shuffled_prediction = predict(shuffled_model, query_x, device)
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(np.sum(fit)),
                "query_count": int(np.sum(query)),
                "mean_baseline": evaluate(mean, query_targets),
                "geometry_manifold": evaluate(current_prediction, query_targets),
                "shuffled_target_control": evaluate(shuffled_prediction, query_targets),
            }
        )
        for index in range(4):
            pooled_truth[index].append(query_targets[index])
            pooled_prediction[index].append(current_prediction[index])
            pooled_mean[index].append(mean[index])
            pooled_shuffle[index].append(shuffled_prediction[index])
        print(
            f"v318 fold={fold} V={fold_rows[-1]['geometry_manifold']['vertical']:.4f} "
            f"UE={fold_rows[-1]['geometry_manifold']['ue_port']:.4f}",
            flush=True,
        )
    truth = tuple(np.concatenate(value) for value in pooled_truth)
    manifold = tuple(np.concatenate(value) for value in pooled_prediction)
    mean = tuple(np.concatenate(value) for value in pooled_mean)
    shuffled = tuple(np.concatenate(value) for value in pooled_shuffle)
    pooled = {
        "mean_baseline": evaluate(mean, truth),
        "geometry_manifold": evaluate(manifold, truth),
        "shuffled_target_control": evaluate(shuffled, truth),
    }
    vertical_wins = sum(
        row["geometry_manifold"]["vertical"] >= row["mean_baseline"]["vertical"] + 0.10
        for row in fold_rows
    )
    ue_wins = sum(
        row["geometry_manifold"]["ue_port"] >= row["mean_baseline"]["ue_port"] + 0.05
        for row in fold_rows
    )
    passed = (
        pooled["geometry_manifold"]["vertical"] >= pooled["mean_baseline"]["vertical"] + 0.10
        and pooled["geometry_manifold"]["ue_port"] >= pooled["mean_baseline"]["ue_port"] + 0.05
        and vertical_wins >= 4
        and ue_wins >= 4
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Official one-bounce geometry predicts per-path BS-V, UE-port and polarization "
            "fingerprints beyond global profile prevalence."
        ),
        "sample_count": int(len(data["point_id"])),
        "fold_rows": fold_rows,
        "pooled": pooled,
        "decision_rule": {
            "vertical_gain": ">=+0.10 and >=4/5 folds",
            "ue_port_gain": ">=+0.05 and >=4/5 folds",
            "passed": bool(passed),
        },
        "outputs": {
            "dataset": PATHS.relative(DATA_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
