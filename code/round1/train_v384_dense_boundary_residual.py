#!/usr/bin/env python3
"""Train a dense-city geometry residual over a support-only local baseline."""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from torch import nn

import build_v382_dense_lifecycle_examples as v382
import lifecycle_geometry_features as lgf
import train_v375_synthetic_lifecycle_pretraining as v375
from project_paths import PATHS


VERSION_NUMBER = 384
VERSION = "v384_dense_boundary_residual"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
EXAMPLES = PATHS.cache_dir(383) / "v383_isolation_matched_lifecycle_examples.npz"
BASELINE_CACHE = OUT_DIR / "v384_support_baseline.npz"
CHECKPOINT = OUT_DIR / "v384_dense_boundary_residual_model.pt"

BASELINE_TREES = 300
MAX_EPOCHS = 40
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-5
MASS_LOSS_WEIGHT = 0.20
CORRECTION_PENALTY = 0.01

MIN_FULL_VS_BASELINE_AP = 0.02
MIN_FULL_VS_BASELINE_MASS = 0.02
MIN_FULL_VS_SHUFFLED_AP = 0.02
MIN_FULL_VS_SHUFFLED_MASS = 0.02
MIN_SCENE_WINS = 4


def support_columns() -> np.ndarray:
    dimensions = lgf.feature_dimensions()
    start = dimensions["point"]
    end = start + dimensions["path"] + dimensions["support"]
    return np.arange(start, end)


def fit_support_baseline(
    train_feature: np.ndarray,
    train_binary: np.ndarray,
    train_mass: np.ndarray,
    query_feature: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    classifier = ExtraTreesClassifier(
        n_estimators=BASELINE_TREES,
        max_depth=18,
        min_samples_leaf=8,
        max_features=0.55,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    classifier.fit(train_feature, train_binary)
    probability = classifier.predict_proba(query_feature)[:, 1].astype(np.float32)
    positive = train_binary.astype(bool)
    regressor = ExtraTreesRegressor(
        n_estimators=BASELINE_TREES,
        max_depth=18,
        min_samples_leaf=8,
        max_features=0.55,
        n_jobs=-1,
        random_state=seed + 10_000,
    )
    regressor.fit(
        train_feature[positive],
        np.log(np.maximum(train_mass[positive], 1e-7)),
    )
    conditional = np.exp(regressor.predict(query_feature)).astype(np.float32)
    return probability, (probability * conditional).astype(np.float32)


def build_baseline(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if BASELINE_CACHE.is_file():
        cached = np.load(BASELINE_CACHE)
        return {key: cached[key] for key in cached.files}
    feature = data["feature"][:, support_columns()].astype(np.float32)
    binary = data["binary"].astype(np.int8)
    mass = data["mass"].astype(np.float32)
    scene = data["scene"].astype(np.int64)
    train_keep = np.isin(scene, v382.TRAIN_SCENES)
    validation_keep = np.isin(scene, v382.VALIDATION_SCENES)
    test_keep = np.isin(scene, v382.TEST_SCENES)
    train_rows = np.flatnonzero(train_keep)
    probability = np.zeros(len(feature), dtype=np.float32)
    mass_score = np.zeros(len(feature), dtype=np.float32)
    for fold in range(5):
        query = train_rows[(scene[train_rows] % 5) == fold]
        fit = np.setdiff1d(train_rows, query, assume_unique=True)
        probability[query], mass_score[query] = fit_support_baseline(
            feature[fit], binary[fit], mass[fit], feature[query], 384_000 + fold
        )
        print(
            f"v384 baseline fold={fold} fit={len(fit)} query={len(query)}",
            flush=True,
        )
    fit = train_rows
    for name, keep, seed in (
        ("validation", validation_keep, 384_100),
        ("test", test_keep, 384_200),
    ):
        query = np.flatnonzero(keep)
        probability[query], mass_score[query] = fit_support_baseline(
            feature[fit], binary[fit], mass[fit], feature[query], seed
        )
        print(f"v384 baseline {name} query={len(query)}", flush=True)
    output = {"probability": probability, "mass": mass_score}
    np.savez_compressed(BASELINE_CACHE, **output)
    return output


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-4, 1.0 - 1e-4)
    return np.log(probability) - np.log1p(-probability)


def baseline_log_conditional(
    probability: np.ndarray, mass: np.ndarray
) -> np.ndarray:
    conditional = np.divide(mass, np.maximum(probability, 1e-5))
    return np.clip(np.log(np.maximum(conditional, 1e-8)), -16.0, 2.0)


def context_features(probability: np.ndarray, mass: np.ndarray) -> np.ndarray:
    logit = np.clip(clipped_logit(probability), -12.0, 12.0)
    log_mass = baseline_log_conditional(probability, mass)
    uncertainty = 4.0 * probability * (1.0 - probability)
    return np.column_stack(
        [logit / 4.0, (log_mass + 6.0) / 4.0, uncertainty]
    ).astype(np.float32)


class BoundaryResidualNet(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.existence = nn.Sequential(nn.Linear(131, 64), nn.SiLU(), nn.Linear(64, 1))
        self.log_mass = nn.Sequential(nn.Linear(131, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.existence[-1].weight)
        nn.init.zeros_(self.existence[-1].bias)
        nn.init.zeros_(self.log_mass[-1].weight)
        nn.init.zeros_(self.log_mass[-1].bias)

    def forward(
        self, feature: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(feature)
        joined = torch.cat([hidden, context], dim=1)
        return self.existence(joined).squeeze(1), self.log_mass(joined).squeeze(1)


def predict(
    model: BoundaryResidualNet,
    feature: np.ndarray,
    probability: np.ndarray,
    mass: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    output_probability = np.empty(len(feature), dtype=np.float32)
    output_log_mass = np.empty(len(feature), dtype=np.float32)
    context = context_features(probability, mass)
    offset_logit = clipped_logit(probability)
    offset_mass = baseline_log_conditional(probability, mass)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(feature), 16384):
            end = min(start + 16384, len(feature))
            feature_batch = torch.from_numpy(feature[start:end]).to(device)
            context_batch = torch.from_numpy(context[start:end]).to(device)
            existence, log_mass = model(feature_batch, context_batch)
            score = offset_logit[start:end] + existence.cpu().numpy()
            output_probability[start:end] = 1.0 / (
                1.0 + np.exp(-np.clip(score, -30.0, 30.0))
            )
            output_log_mass[start:end] = np.clip(
                offset_mass[start:end] + log_mass.cpu().numpy(), -16.0, 2.0
            )
    return output_probability, output_log_mass


def train_residual(
    *,
    name: str,
    feature: np.ndarray,
    binary: np.ndarray,
    mass: np.ndarray,
    group: np.ndarray,
    baseline_probability: np.ndarray,
    baseline_mass: np.ndarray,
    train_keep: np.ndarray,
    validation_keep: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[BoundaryResidualNet, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BoundaryResidualNet(feature.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_rows = np.flatnonzero(train_keep)
    validation_rows = np.flatnonzero(validation_keep)
    positive_count = max(int(np.sum(binary[train_rows])), 1)
    negative_count = max(len(train_rows) - positive_count, 1)
    positive_weight = torch.tensor(
        min(negative_count / positive_count, 8.0), device=device
    )
    context = context_features(baseline_probability, baseline_mass)
    offset_logit = clipped_logit(baseline_probability).astype(np.float32)
    offset_mass = baseline_log_conditional(
        baseline_probability, baseline_mass
    ).astype(np.float32)
    rng = np.random.default_rng(seed)
    best_state = None
    best_objective = -np.inf
    selected_epoch = 0
    epoch_rows = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            feature_batch = torch.from_numpy(feature[rows]).to(device)
            context_batch = torch.from_numpy(context[rows]).to(device)
            binary_batch = torch.from_numpy(binary[rows].astype(np.float32)).to(device)
            mass_batch = torch.from_numpy(mass[rows]).to(device)
            offset_logit_batch = torch.from_numpy(offset_logit[rows]).to(device)
            offset_mass_batch = torch.from_numpy(offset_mass[rows]).to(device)
            existence, log_mass = model(feature_batch, context_batch)
            score = offset_logit_batch + existence
            existence_loss = nn.functional.binary_cross_entropy_with_logits(
                score,
                binary_batch,
                pos_weight=positive_weight,
                reduction="none",
            )
            uncertainty = 4.0 * torch.sigmoid(offset_logit_batch) * (
                1.0 - torch.sigmoid(offset_logit_batch)
            )
            existence_loss = torch.mean(existence_loss * (0.35 + 2.0 * uncertainty))
            positive = binary_batch > 0.5
            if torch.any(positive):
                target = torch.log(torch.clamp(mass_batch[positive], min=1e-7))
                mass_loss = nn.functional.smooth_l1_loss(
                    offset_mass_batch[positive] + log_mass[positive], target
                )
            else:
                mass_loss = existence_loss * 0.0
            penalty = torch.mean(existence**2) + torch.mean(log_mass**2)
            loss = (
                existence_loss
                + MASS_LOSS_WEIGHT * mass_loss
                + CORRECTION_PENALTY * penalty
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability, log_mass = predict(
            model,
            feature[validation_rows],
            baseline_probability[validation_rows],
            baseline_mass[validation_rows],
            device,
        )
        validation = v375.metrics(
            binary[validation_rows],
            mass[validation_rows],
            group[validation_rows],
            probability,
            log_mass,
        )
        objective = (
            0.60 * validation["existence_ap"]
            + 0.40 * validation["group_mass_cosine"]
        )
        epoch_rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "selection_objective": float(objective),
                **validation,
            }
        )
        if objective > best_objective:
            best_objective = objective
            selected_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"v384 {name} epoch={epoch} loss={np.mean(losses):.4f} "
                f"val_ap={validation['existence_ap']:.4f} "
                f"mass={validation['group_mass_cosine']:.4f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("v384 selected no residual checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "selected_epoch": selected_epoch,
        "best_validation_objective": float(best_objective),
        "epoch_rows": epoch_rows,
    }


def evaluate(
    model: BoundaryResidualNet,
    feature: np.ndarray,
    data: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    keep: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    rows = np.flatnonzero(keep)
    probability, log_mass = predict(
        model,
        feature[rows],
        baseline["probability"][rows],
        baseline["mass"][rows],
        device,
    )
    metrics = v375.metrics(
        data["binary"][rows],
        data["mass"][rows],
        data["group"][rows],
        probability,
        log_mass,
    )
    return metrics, probability, log_mass


def baseline_metrics(
    data: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    keep: np.ndarray,
) -> dict[str, float]:
    rows = np.flatnonzero(keep)
    log_mass = baseline_log_conditional(
        baseline["probability"][rows], baseline["mass"][rows]
    )
    return v375.metrics(
        data["binary"][rows],
        data["mass"][rows],
        data["group"][rows],
        baseline["probability"][rows],
        log_mass,
    )


def main() -> None:
    if not EXAMPLES.is_file():
        raise FileNotFoundError(EXAMPLES)
    cached = np.load(EXAMPLES)
    data = {
        key: cached[key]
        for key in ("feature", "binary", "mass", "scene", "family", "group")
    }
    data["feature"] = data["feature"].astype(np.float32)
    data["binary"] = data["binary"].astype(np.int8)
    data["mass"] = data["mass"].astype(np.float32)
    scene = data["scene"].astype(np.int64)
    train_keep = np.isin(scene, v382.TRAIN_SCENES)
    validation_keep = np.isin(scene, v382.VALIDATION_SCENES)
    test_keep = np.isin(scene, v382.TEST_SCENES)
    baseline = build_baseline(data)

    mean = np.mean(data["feature"][train_keep], axis=0).astype(np.float32)
    std = np.maximum(
        np.std(data["feature"][train_keep], axis=0).astype(np.float32), 1e-4
    )
    normalized = np.clip((data["feature"] - mean) / std, -8.0, 8.0).astype(
        np.float32
    )
    shuffled = v375.shuffled_geometry(normalized, scene, 384_300)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model, full_training = train_residual(
        name="full",
        feature=normalized,
        binary=data["binary"],
        mass=data["mass"],
        group=data["group"],
        baseline_probability=baseline["probability"],
        baseline_mass=baseline["mass"],
        train_keep=train_keep,
        validation_keep=validation_keep,
        device=device,
        seed=38401,
    )
    shuffled_model, shuffled_training = train_residual(
        name="shuffled_geometry",
        feature=shuffled,
        binary=data["binary"],
        mass=data["mass"],
        group=data["group"],
        baseline_probability=baseline["probability"],
        baseline_mass=baseline["mass"],
        train_keep=train_keep,
        validation_keep=validation_keep,
        device=device,
        seed=38402,
    )
    baseline_test = baseline_metrics(data, baseline, test_keep)
    full_test, full_probability, full_log_mass = evaluate(
        full_model, normalized, data, baseline, test_keep, device
    )
    shuffled_test, _, _ = evaluate(
        shuffled_model, shuffled, data, baseline, test_keep, device
    )
    scene_rows = []
    test_rows = np.flatnonzero(test_keep)
    for scene_id in v382.TEST_SCENES:
        local_keep = scene[test_rows] == scene_id
        rows = test_rows[local_keep]
        base = baseline_metrics(data, baseline, np.isin(np.arange(len(scene)), rows))
        full = v375.metrics(
            data["binary"][rows],
            data["mass"][rows],
            data["group"][rows],
            full_probability[local_keep],
            full_log_mass[local_keep],
        )
        scene_rows.append(
            {
                "scene": int(scene_id),
                "baseline": base,
                "full": full,
                "ap_delta": float(full["existence_ap"] - base["existence_ap"]),
                "mass_delta": float(
                    full["group_mass_cosine"] - base["group_mass_cosine"]
                ),
            }
        )
    deltas = {
        "full_vs_baseline_ap": float(
            full_test["existence_ap"] - baseline_test["existence_ap"]
        ),
        "full_vs_baseline_mass": float(
            full_test["group_mass_cosine"] - baseline_test["group_mass_cosine"]
        ),
        "full_vs_shuffled_ap": float(
            full_test["existence_ap"] - shuffled_test["existence_ap"]
        ),
        "full_vs_shuffled_mass": float(
            full_test["group_mass_cosine"] - shuffled_test["group_mass_cosine"]
        ),
        "full_vs_baseline_log_mass_rmse_reduction": float(
            baseline_test["conditional_log_mass_rmse"]
            - full_test["conditional_log_mass_rmse"]
        ),
    }
    scene_wins = int(
        sum(row["ap_delta"] > 0.0 and row["mass_delta"] > 0.0 for row in scene_rows)
    )
    passed = bool(
        deltas["full_vs_baseline_ap"] >= MIN_FULL_VS_BASELINE_AP
        and deltas["full_vs_baseline_mass"] >= MIN_FULL_VS_BASELINE_MASS
        and deltas["full_vs_shuffled_ap"] >= MIN_FULL_VS_SHUFFLED_AP
        and deltas["full_vs_shuffled_mass"] >= MIN_FULL_VS_SHUFFLED_MASS
        and scene_wins >= MIN_SCENE_WINS
    )
    torch.save(
        {
            "state_dict": full_model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "source_examples": PATHS.relative(EXAMPLES),
            "selected_epoch": full_training["selected_epoch"],
            "parameters": {
                "feature_dimension": int(data["feature"].shape[1]),
                "baseline_columns": support_columns(),
                "model": "local-offset boundary residual",
            },
        },
        CHECKPOINT,
    )
    report = {
        "method": VERSION,
        "status": "synthetic_residual_passed" if passed else "analysis_only_no_transfer",
        "scope": "unseen dense-scene boundary residual pretraining",
        "primary_hypothesis": (
            "Point/ray geometry predicts the official-matched local support baseline's "
            "existence and conditional-mass residual on unseen dense cities."
        ),
        "parameters": {
            "baseline_trees_per_head": BASELINE_TREES,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "mass_loss_weight": MASS_LOSS_WEIGHT,
            "correction_penalty": CORRECTION_PENALTY,
        },
        "training": {
            "full": full_training,
            "shuffled_geometry": shuffled_training,
        },
        "test": {
            "baseline": baseline_test,
            "full": full_test,
            "shuffled_geometry": shuffled_test,
            "deltas": deltas,
            "scene_rows": scene_rows,
            "scene_wins": scene_wins,
        },
        "decision_rule": {
            "full_vs_baseline_ap": f">={MIN_FULL_VS_BASELINE_AP}",
            "full_vs_baseline_mass": f">={MIN_FULL_VS_BASELINE_MASS}",
            "full_vs_shuffled_ap": f">={MIN_FULL_VS_SHUFFLED_AP}",
            "full_vs_shuffled_mass": f">={MIN_FULL_VS_SHUFFLED_MASS}",
            "scene_wins": f">={MIN_SCENE_WINS}/4",
            "passed": passed,
        },
        "outputs": {
            "baseline_cache": PATHS.relative(BASELINE_CACHE),
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
