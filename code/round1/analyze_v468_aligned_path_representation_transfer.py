#!/usr/bin/env python3
"""Transfer an aligned synthetic path-event representation to official folds."""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score

import analyze_v464_multiconfig_path_event_feasibility as v464
import analyze_v467_synthetic_pointcloud_candidate_prior as v467
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 468
VERSION = "v468_aligned_path_representation_transfer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
EMBEDDING_CACHE = OUT_DIR / "v468_official_path_embedding.npy"

EMBEDDING_DIM = 16
HIDDEN = 128
EPOCHS = 45
EVAL_EVERY = 5
BATCH_SIZE = 4096
LEARNING_RATE = 2e-3
MASS_LOSS_WEIGHT = 0.20
MIN_OFFICIAL_AP_GAIN = 0.01
MIN_OFFICIAL_FOLD_WINS = 7


class PathEventNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.SiLU(),
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, EMBEDDING_DIM),
        )
        self.exist = nn.Linear(EMBEDDING_DIM, 1)
        self.mass = nn.Linear(EMBEDDING_DIM, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.backbone(value)
        normalized = F.normalize(embedding, dim=1)
        return self.exist(normalized).squeeze(1), self.mass(normalized).squeeze(1), normalized


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardized(
    feature: np.ndarray, fit_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(feature[fit_rows], axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(feature[fit_rows], axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return ((feature - mean) / std).astype(np.float32), mean, std


@torch.no_grad()
def predict(
    model: PathEventNet,
    feature: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.empty(len(feature), dtype=np.float32)
    log_mass = np.empty(len(feature), dtype=np.float32)
    embedding = np.empty((len(feature), EMBEDDING_DIM), dtype=np.float32)
    model.eval()
    for start in range(0, len(feature), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(feature))
        x = torch.from_numpy(feature[start:end]).to(device)
        existence, mass, hidden = model(x)
        probability[start:end] = torch.sigmoid(existence).cpu().numpy()
        log_mass[start:end] = mass.cpu().numpy()
        embedding[start:end] = hidden.cpu().numpy()
    return probability, log_mass, embedding


def train_fold(
    feature: np.ndarray,
    label: np.ndarray,
    mass: np.ndarray,
    fit_rows: np.ndarray,
    query_rows: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[PathEventNet, np.ndarray, np.ndarray, dict[str, object]]:
    set_seed(seed)
    normalized, mean, std = standardized(feature, fit_rows)
    model = PathEventNet(feature.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=2e-4
    )
    fit_label = label[fit_rows].astype(np.float32)
    positive_count = max(float(np.sum(fit_label)), 1.0)
    negative_count = max(float(len(fit_label) - np.sum(fit_label)), 1.0)
    positive_weight = torch.tensor(
        np.sqrt(negative_count / positive_count), dtype=torch.float32, device=device
    )
    positive_log_mass = np.log(np.maximum(mass[fit_rows][fit_label > 0], 1e-6))
    mass_mean = float(np.mean(positive_log_mass))
    mass_std = float(max(np.std(positive_log_mass), 0.25))
    rng = np.random.default_rng(seed)
    history = []
    best_ap = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(fit_rows)
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            x = torch.from_numpy(normalized[rows]).to(device)
            y = torch.from_numpy(label[rows].astype(np.float32)).to(device)
            target_mass = torch.from_numpy(
                ((np.log(np.maximum(mass[rows], 1e-6)) - mass_mean) / mass_std).astype(
                    np.float32
                )
            ).to(device)
            existence, predicted_mass, _embedding = model(x)
            existence_loss = F.binary_cross_entropy_with_logits(
                existence, y, pos_weight=positive_weight
            )
            positive = y > 0.5
            if torch.any(positive):
                mass_loss = F.smooth_l1_loss(
                    predicted_mass[positive], target_mass[positive]
                )
            else:
                mass_loss = existence_loss.new_zeros(())
            loss = existence_loss + MASS_LOSS_WEIGHT * mass_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            probability, predicted_mass, _embedding = predict(
                model, normalized[query_rows], device
            )
            ap = float(average_precision_score(label[query_rows], probability))
            positive = label[query_rows]
            mass_correlation = 0.0
            if np.sum(positive) >= 3:
                mass_correlation = float(
                    np.corrcoef(
                        predicted_mass[positive],
                        np.log(np.maximum(mass[query_rows][positive], 1e-6)),
                    )[0, 1]
                )
            history.append(
                {
                    "epoch": epoch,
                    "average_precision": ap,
                    "positive_log_mass_correlation": mass_correlation,
                }
            )
            if ap > best_ap:
                best_ap = ap
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        print(
            f"v468 seed={seed} epoch={epoch}/{EPOCHS} best_ap={best_ap:.4f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    return model, mean, std, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_average_precision": float(best_ap),
        "mass_mean": mass_mean,
        "mass_std": mass_std,
        "history": history,
    }


def official_validation(
    base_feature: np.ndarray,
    transfer_feature: np.ndarray,
    point_id: np.ndarray,
    label: np.ndarray,
    mass: np.ndarray,
    point_groups: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    rows = []
    gains = []
    wins = 0
    for split_name, groups in point_groups.items():
        for fold in range(5):
            query = groups[point_id] == fold
            fit = ~query
            calibration_ids = v467.calibration_points(
                point_id[fit], fold + (0 if split_name == "region" else 10)
            )
            calibration = fit & np.isin(point_id, calibration_ids)
            train = fit & (~calibration)
            train_rows = v464.choose_training_rows(train, label, 468_100 + fold)
            variants = {
                "baseline": base_feature,
                "representation": np.column_stack([base_feature, transfer_feature]),
            }
            result = {}
            for variant_name, feature in variants.items():
                model = v467.fit_model(
                    feature,
                    label,
                    mass,
                    train_rows,
                    468_100 + fold + (100 if variant_name == "representation" else 0),
                )
                calibration_probability = model.predict_proba(feature[calibration])[:, 1]
                threshold = v464.precision_threshold(
                    calibration_probability, label[calibration], mass[calibration]
                )
                probability = model.predict_proba(feature[query])[:, 1]
                result[variant_name] = v464.evaluate_prediction(
                    probability, label[query], mass[query], threshold
                )
            gain = (
                result["representation"]["average_precision"]
                - result["baseline"]["average_precision"]
            )
            gains.append(gain)
            wins += int(gain > 0.0)
            rows.append(
                {
                    "split": split_name,
                    "fold": fold,
                    "baseline": result["baseline"],
                    "representation": result["representation"],
                    "average_precision_gain": gain,
                }
            )
            print(
                f"v468 official {split_name} fold={fold} AP gain={gain:+.5f}",
                flush=True,
            )
    summary = {
        "mean_average_precision_gain": float(np.mean(gains)),
        "region_average_precision_gain": float(
            np.mean([row["average_precision_gain"] for row in rows if row["split"] == "region"])
        ),
        "mode_average_precision_gain": float(
            np.mean([row["average_precision_gain"] for row in rows if row["split"] == "mode"])
        ),
        "fold_wins": int(wins),
    }
    return rows, summary


def main() -> None:
    for path in (v467.CANDIDATE_CACHE, v467.LABEL_CACHE, v467.V464_DATA, v467.OFFICIAL_PATHS):
        v467.require_file(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synthetic_data = np.load(v467.CANDIDATE_CACHE)
    synthetic_paths = {key: synthetic_data[key] for key in synthetic_data.files}
    labels = np.load(v467.LABEL_CACHE)
    synthetic_point, synthetic_candidate, synthetic_feature = v467.path_features(
        synthetic_paths, ""
    )
    synthetic_mass = labels["label"][synthetic_point, synthetic_candidate].astype(np.float32)
    synthetic_label = synthetic_mass > 0.0
    groups = KMeans(n_clusters=5, random_state=468, n_init=20).fit_predict(
        synthetic_paths["pos"][:, :2]
    )

    official_path_data = np.load(v467.OFFICIAL_PATHS)
    official_paths = {key: official_path_data[key] for key in official_path_data.files}
    official_point, official_candidate, official_feature = v467.path_features(
        official_paths, "train_"
    )
    official_embeddings = []
    official_probability = []
    official_log_mass = []
    fold_reports = []

    for fold in range(5):
        query_rows = np.flatnonzero(groups[synthetic_point] == fold)
        fit_rows = np.flatnonzero(groups[synthetic_point] != fold)
        model, mean, std, fold_report = train_fold(
            synthetic_feature,
            synthetic_label,
            synthetic_mass,
            fit_rows,
            query_rows,
            468_000 + fold,
            device,
        )
        official_normalized = ((official_feature - mean) / std).astype(np.float32)
        probability, log_mass, embedding = predict(model, official_normalized, device)
        official_probability.append(probability)
        official_log_mass.append(log_mass)
        official_embeddings.append(embedding)
        fold_reports.append(fold_report)

    probability_stack = np.stack(official_probability, axis=1)
    mass_stack = np.stack(official_log_mass, axis=1)
    embedding_stack = np.stack(official_embeddings, axis=1)
    transfer_rows = np.concatenate(
        [
            np.mean(embedding_stack, axis=1),
            np.std(embedding_stack, axis=1),
            np.mean(probability_stack, axis=1, keepdims=True),
            np.std(probability_stack, axis=1, keepdims=True),
            np.mean(mass_stack, axis=1, keepdims=True),
            np.std(mass_stack, axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32)
    transfer_matrix = np.zeros((2000, v467.PATH_LIMIT, transfer_rows.shape[1]), dtype=np.float32)
    transfer_matrix[official_point, official_candidate] = transfer_rows

    official_data = np.load(v467.V464_DATA)
    point_id = official_data["point_id"].astype(np.int32)
    candidate_id = official_data["candidate_id"].astype(np.int16)
    base_dim = int(official_data["base_dim"])
    base_feature = official_data["feature"][:, :base_dim].astype(np.float32)
    official_label = official_data["label"].astype(bool)
    official_mass = official_data["mass"].astype(np.float32)
    transfer_feature = transfer_matrix[point_id, candidate_id]
    np.save(EMBEDDING_CACHE, transfer_feature.astype(np.float32))

    path_a = np.load(v464.PATH_A)
    path_b = np.load(v464.PATH_B)
    h_a = v464.candidate_h(path_a, "train")
    h_b = v464.candidate_h(path_b, "train")
    stability = v464.multiconfig_stability(path_a, path_b, h_a, h_b)
    point_groups = v464.point_fold_labels(path_a, stability)
    official_rows, official_summary = official_validation(
        base_feature,
        transfer_feature,
        point_id,
        official_label,
        official_mass,
        point_groups,
    )
    passed = (
        official_summary["region_average_precision_gain"] >= MIN_OFFICIAL_AP_GAIN
        and official_summary["mode_average_precision_gain"] >= MIN_OFFICIAL_AP_GAIN
        and official_summary["fold_wins"] >= MIN_OFFICIAL_FOLD_WINS
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The aligned synthetic candidate model contains transferable propagation structure "
            "that is lost when v467 compresses it to one scalar probability."
        ),
        "compliance_boundary": {
            "uses": [
                "v467 aligned random-position synthetic candidate labels",
                "official point-cloud path candidates",
                "official train-H labels only within each validation fold",
            ],
            "does_not_use": [
                "Sionna at official test coordinates",
                "hidden test H or online scores",
                "v212/v458 test teacher outputs",
            ],
        },
        "parameters": {
            "device": str(device),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "mass_loss_weight": MASS_LOSS_WEIGHT,
            "synthetic_models": 5,
            "transfer_feature_dim": int(transfer_feature.shape[1]),
        },
        "synthetic_fold_reports": fold_reports,
        "official_validation": {
            "fold_rows": official_rows,
            "summary": official_summary,
            "transfer_feature": summarize(transfer_feature),
        },
        "decision_rule": {
            "official_region_average_precision_gain": f">={MIN_OFFICIAL_AP_GAIN}",
            "official_mode_average_precision_gain": f">={MIN_OFFICIAL_AP_GAIN}",
            "official_fold_wins": f">={MIN_OFFICIAL_FOLD_WINS}/10",
            "passed": bool(passed),
        },
        "next_action": (
            "Scale the aligned candidate dataset and train a set encoder only if this richer "
            "representation passes; otherwise the remaining mismatch is candidate geometry, not capacity."
        ),
        "outputs": {
            "embedding": PATHS.relative(EMBEDDING_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"official": official_summary, "passed": passed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
