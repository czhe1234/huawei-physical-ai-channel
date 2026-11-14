#!/usr/bin/env python3
"""Confirm BS vertical-response learning on entirely unseen wall patches."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v318_geometry_array_fingerprint_manifold as v318
from project_paths import PATHS


VERSION_NUMBER = 319
VERSION = "v319_unseen_patch_vertical_manifold"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
FOLD_COUNT = 5
EPOCHS = 100
BATCH_SIZE = 256


class VerticalMLP(nn.Module):
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
            nn.Linear(96, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(x), dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_folds(patch_id: np.ndarray) -> list[np.ndarray]:
    unique, count = np.unique(patch_id, return_counts=True)
    order = np.argsort(count)[::-1]
    fold_patch: list[list[int]] = [[] for _ in range(FOLD_COUNT)]
    fold_count = np.zeros(FOLD_COUNT, dtype=np.int64)
    for index in order:
        fold = int(np.argmin(fold_count))
        fold_patch[fold].append(int(unique[index]))
        fold_count[fold] += int(count[index])
    return [np.asarray(value, dtype=np.int64) for value in fold_patch]


def train(
    x: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    seed: int,
    device: torch.device,
) -> VerticalMLP:
    set_seed(seed)
    model = VerticalMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    x_tensor = torch.from_numpy(x).to(device)
    target_tensor = torch.from_numpy(target).to(device)
    weight_tensor = torch.from_numpy(weight).to(device)
    rng = np.random.default_rng(seed)
    for _epoch in range(EPOCHS):
        order = rng.permutation(len(x))
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            ids = torch.from_numpy(order[start : start + BATCH_SIZE]).to(device)
            prediction = model(x_tensor[ids])
            per_sample = 1.0 - F.cosine_similarity(prediction, target_tensor[ids], dim=1)
            loss = torch.sum(per_sample * weight_tensor[ids]) / torch.sum(weight_tensor[ids]).clamp_min(1e-20)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict(model: VerticalMLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(x).to(device)).cpu().numpy()


def cosine(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(v318.profile_cosine(prediction, target)))


def main() -> None:
    data = v318.build_dataset()
    folds = patch_folds(data["patch_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    all_truth = []
    all_mean = []
    all_prediction = []
    all_shuffle = []
    for fold, query_patches in enumerate(folds):
        query = np.isin(data["patch_id"], query_patches)
        fit = ~query
        mean_profile = np.average(
            data["vertical"][fit], axis=0, weights=data["confidence"][fit]
        )
        mean = np.broadcast_to(mean_profile, data["vertical"][query].shape).copy().astype(np.float32)
        x_mean = data["numeric"][fit].mean(axis=0)
        x_std = data["numeric"][fit].std(axis=0)
        x_std[x_std < 1e-5] = 1.0
        fit_x = ((data["numeric"][fit] - x_mean) / x_std).astype(np.float32)
        query_x = ((data["numeric"][query] - x_mean) / x_std).astype(np.float32)
        weight = (
            np.sqrt(np.maximum(data["mass"][fit], 1e-5)) * data["confidence"][fit]
        ).astype(np.float32)
        current = train(fit_x, data["vertical"][fit], weight, 319 + fold, device)
        prediction = predict(current, query_x, device)
        rng = np.random.default_rng(31_900 + fold)
        shuffled_target = data["vertical"][fit][rng.permutation(np.sum(fit))]
        shuffled_model = train(fit_x, shuffled_target, weight, 1319 + fold, device)
        shuffled = predict(shuffled_model, query_x, device)
        truth = data["vertical"][query]
        rows.append(
            {
                "fold": fold,
                "fit_patch_count": int(len(np.unique(data["patch_id"][fit]))),
                "query_patch_count": int(len(query_patches)),
                "fit_sample_count": int(np.sum(fit)),
                "query_sample_count": int(np.sum(query)),
                "mean_baseline": cosine(mean, truth),
                "geometry_manifold": cosine(prediction, truth),
                "shuffled_target_control": cosine(shuffled, truth),
            }
        )
        all_truth.append(truth)
        all_mean.append(mean)
        all_prediction.append(prediction)
        all_shuffle.append(shuffled)
        print(
            f"v319 fold={fold} patches={len(query_patches)} "
            f"mean={rows[-1]['mean_baseline']:.4f} model={rows[-1]['geometry_manifold']:.4f}",
            flush=True,
        )
    truth = np.concatenate(all_truth)
    mean = np.concatenate(all_mean)
    prediction = np.concatenate(all_prediction)
    shuffled = np.concatenate(all_shuffle)
    pooled = {
        "mean_baseline": cosine(mean, truth),
        "geometry_manifold": cosine(prediction, truth),
        "shuffled_target_control": cosine(shuffled, truth),
    }
    gain = pooled["geometry_manifold"] - pooled["mean_baseline"]
    wins = int(
        sum(row["geometry_manifold"] >= row["mean_baseline"] + 0.25 for row in rows)
    )
    passed = gain >= 0.25 and wins >= 4
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The BS vertical fingerprint is a geometry-dependent array manifold that "
            "generalizes to wall patches absent from training."
        ),
        "fold_design": "All samples from each wall patch belong to exactly one query fold.",
        "fold_rows": rows,
        "pooled": pooled | {"gain_over_mean": gain, "fold_wins_at_0p25": wins},
        "decision_rule": {
            "pooled_gain_over_mean": ">=0.25",
            "fold_wins_at_0p25": ">=4/5",
            "passed": bool(passed),
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
