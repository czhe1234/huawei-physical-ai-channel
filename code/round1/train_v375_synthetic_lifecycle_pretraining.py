#!/usr/bin/env python3
"""Train a scene-held-out path lifecycle model on the v374 dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from scipy.spatial import cKDTree
from torch import nn

import analyze_v366_local_eikonal_pdp as v366
import analyze_v372_virtual_source_gap_reconnection as v372
import lifecycle_geometry_features as lgf
from project_paths import PATHS


VERSION_NUMBER = 375
VERSION = "v375_synthetic_lifecycle_pretraining"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SCENE_ROOT = PATHS.cache_dir(374) / "scenes"
EXAMPLES = OUT_DIR / "v375_synthetic_lifecycle_examples.npz"
CHECKPOINT = OUT_DIR / "v375_synthetic_lifecycle_model.pt"
REPORT = REPORT_DIR / f"{VERSION}.json"

FAMILIES_PER_SCENE = 240
MAX_POSITIVE_PER_FOLD = 18
MAX_NEGATIVE_PER_FOLD = 30
MAX_NEGATIVE_DISTANCE = 130.0
TRAIN_SCENES = tuple(range(18))
VALIDATION_SCENES = (18, 19, 20)
TEST_SCENES = (21, 22, 23)
MAX_EPOCHS = 45
BATCH_SIZE = 4096
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 2e-5

MIN_TEST_AP = 0.70
MIN_AP_CONTROL_LIFT = 0.05
MIN_MASS_CONTROL_LIFT = 0.03
MIN_SCENE_WINS = 3


def spatial_fold(position: np.ndarray) -> np.ndarray:
    median_x = float(np.median(position[:, 0]))
    median_y = float(np.median(position[:, 1]))
    return (
        (position[:, 0] >= median_x).astype(np.int8)
        + 2 * (position[:, 1] >= median_y).astype(np.int8)
    )


def fit_family_surface(
    position: np.ndarray,
    delay: np.ndarray,
    mass: np.ndarray,
) -> tuple[v372.AffineSurface, float]:
    model = v372.fit_affine_surface(position[:, :2], delay, mass)
    predicted, _ = v372.affine_prediction(model, position[:, :2])
    error = v366.signed_circular_delta(predicted, delay)
    weight = np.sqrt(np.maximum(mass, 1e-7))
    rmse = float(np.sqrt(np.sum(weight * error**2) / np.sum(weight)))
    return model, rmse


def select_examples(
    *,
    rng: np.random.Generator,
    query_ids: np.ndarray,
    valid: np.ndarray,
    position: np.ndarray,
    support_xy: np.ndarray,
) -> np.ndarray:
    positive = query_ids[valid[query_ids]]
    negative = query_ids[~valid[query_ids]]
    if len(positive) > MAX_POSITIVE_PER_FOLD:
        positive = rng.choice(
            positive, size=MAX_POSITIVE_PER_FOLD, replace=False
        )
    if len(negative):
        distance, _ = cKDTree(support_xy).query(position[negative, :2], k=1)
        near = negative[distance <= MAX_NEGATIVE_DISTANCE]
        far = negative[distance > MAX_NEGATIVE_DISTANCE]
        target = min(
            MAX_NEGATIVE_PER_FOLD,
            max(8, 2 * max(len(positive), 1)),
        )
        near_take = min(len(near), (target + 1) // 2)
        far_take = min(len(far), target - near_take)
        parts = []
        if near_take:
            nearest = np.argsort(distance[distance <= MAX_NEGATIVE_DISTANCE])[:near_take]
            parts.append(near[nearest])
        if far_take:
            parts.append(rng.choice(far, size=far_take, replace=False))
        remaining = target - sum(len(part) for part in parts)
        if remaining > 0:
            used = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            available = np.setdiff1d(negative, used)
            if len(available):
                parts.append(
                    rng.choice(
                        available, size=min(remaining, len(available)), replace=False
                    )
                )
        negative = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    return np.concatenate([positive, negative]).astype(np.int64)


def build_examples() -> dict[str, np.ndarray]:
    if EXAMPLES.is_file():
        cached = np.load(EXAMPLES)
        return {key: cached[key] for key in cached.files}
    rng = np.random.default_rng(37501)
    rows: dict[str, list[np.ndarray]] = {
        "feature": [],
        "binary": [],
        "mass": [],
        "scene": [],
        "family": [],
        "group": [],
    }
    dimensions = lgf.feature_dimensions()
    for scene_id in range(24):
        path = SCENE_ROOT / f"scene_{scene_id:03d}" / "lifecycle.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.load(path)
        position = data["positions"].astype(np.float64)
        bs = data["bs"].astype(np.float64)
        valid = data["valid"].astype(bool)
        delay = data["delay_bin"].astype(np.float64)
        mass = data["normalized_power"].astype(np.float64)
        frequency = float(data["frequency"])
        raster = lgf.raster_from_boxes(data["boxes"].astype(np.float64))
        point_feature = lgf.point_features(raster, position, bs)
        fold = spatial_fold(position)
        family_score = np.sum(mass, axis=0) * np.sqrt(np.maximum(np.sum(valid, axis=0), 1))
        family_ids = np.argsort(family_score)[::-1][:FAMILIES_PER_SCENE]
        scene_feature = []
        scene_binary = []
        scene_mass = []
        scene_family = []
        scene_group = []
        for family_id in family_ids:
            if int(np.sum(valid[:, family_id])) < 8:
                continue
            for held_fold in range(4):
                support = np.flatnonzero(valid[:, family_id] & (fold != held_fold))
                query = np.flatnonzero(fold == held_fold)
                if len(support) < 4 or len(query) == 0:
                    continue
                support_mass = mass[support, family_id]
                model, fit_rmse = fit_family_surface(
                    position[support], delay[support, family_id], support_mass
                )
                selected = select_examples(
                    rng=rng,
                    query_ids=query,
                    valid=valid[:, family_id],
                    position=position,
                    support_xy=position[support, :2],
                )
                if len(selected) == 0:
                    continue
                predicted_delay, predicted_gradient = v372.affine_prediction(
                    model, position[selected, :2]
                )
                group_id = scene_id * 1_000_000 + int(family_id) * 4 + held_fold
                for row, point_id in enumerate(selected):
                    scene_feature.append(
                        lgf.family_feature(
                            scene=raster,
                            point=position[point_id],
                            bs=bs,
                            point_feature=point_feature[point_id],
                            predicted_delay=float(predicted_delay[row]),
                            predicted_gradient=predicted_gradient[row],
                            support_xy=position[support, :2],
                            support_mass=support_mass,
                            fit_delay_rmse=fit_rmse,
                            frequency=frequency,
                        )
                    )
                    scene_binary.append(bool(valid[point_id, family_id]))
                    scene_mass.append(float(mass[point_id, family_id]))
                    scene_family.append(int(family_id))
                    scene_group.append(group_id)
        if not scene_feature:
            raise RuntimeError(f"scene {scene_id} generated no lifecycle examples")
        rows["feature"].append(np.stack(scene_feature).astype(np.float32))
        rows["binary"].append(np.asarray(scene_binary, dtype=np.int8))
        rows["mass"].append(np.asarray(scene_mass, dtype=np.float32))
        rows["scene"].append(np.full(len(scene_binary), scene_id, dtype=np.int8))
        rows["family"].append(np.asarray(scene_family, dtype=np.int32))
        rows["group"].append(np.asarray(scene_group, dtype=np.int64))
        print(
            f"v375 examples scene={scene_id} count={len(scene_binary)} "
            f"positive={int(np.sum(scene_binary))} feature_dim={dimensions['total']}",
            flush=True,
        )
    output = {key: np.concatenate(value) for key, value in rows.items()}
    if output["feature"].shape[1] != dimensions["total"]:
        raise RuntimeError("lifecycle feature dimension mismatch")
    np.savez_compressed(EXAMPLES, **output)
    return output


class LifecycleNet(nn.Module):
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
        self.existence = nn.Linear(128, 1)
        self.log_mass = nn.Linear(128, 1)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(feature)
        return self.existence(hidden).squeeze(1), self.log_mass(hidden).squeeze(1)


def shuffled_geometry(
    feature: np.ndarray,
    scene: np.ndarray,
    seed: int,
) -> np.ndarray:
    output = feature.copy()
    dimensions = lgf.feature_dimensions()
    geometry_columns = np.r_[
        np.arange(dimensions["point"]),
        np.arange(
            dimensions["point"] + dimensions["path"] + dimensions["support"],
            dimensions["total"],
        ),
    ]
    rng = np.random.default_rng(seed)
    for scene_id in np.unique(scene):
        rows = np.flatnonzero(scene == scene_id)
        order = rng.permutation(len(rows))
        output[np.ix_(rows, geometry_columns)] = feature[np.ix_(rows[order], geometry_columns)]
    return output


def predict(
    model: LifecycleNet,
    feature: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.empty(len(feature), dtype=np.float32)
    log_mass = np.empty(len(feature), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(feature), 16384):
            end = min(start + 16384, len(feature))
            batch = torch.from_numpy(feature[start:end]).to(device)
            logit, mass = model(batch)
            probability[start:end] = torch.sigmoid(logit).cpu().numpy()
            log_mass[start:end] = mass.cpu().numpy()
    return probability, log_mass


def group_mass_cosine(
    group: np.ndarray,
    truth_mass: np.ndarray,
    probability: np.ndarray,
    log_mass: np.ndarray,
) -> float:
    prediction = probability * np.exp(np.clip(log_mass, -16.0, 2.0))
    values = []
    for group_id in np.unique(group):
        keep = group == group_id
        truth = truth_mass[keep]
        if float(np.sum(truth)) <= 1e-20:
            continue
        pred = prediction[keep]
        values.append(
            float(
                np.dot(pred, truth)
                / max(float(np.linalg.norm(pred) * np.linalg.norm(truth)), 1e-30)
            )
        )
    return float(np.mean(values)) if values else float("nan")


def metrics(
    binary: np.ndarray,
    mass: np.ndarray,
    group: np.ndarray,
    probability: np.ndarray,
    log_mass: np.ndarray,
) -> dict[str, float]:
    positive = binary.astype(bool)
    return {
        "existence_ap": float(average_precision_score(binary, probability)),
        "conditional_log_mass_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        log_mass[positive]
                        - np.log(np.maximum(mass[positive], 1e-7))
                    )
                    ** 2
                )
            )
        ),
        "group_mass_cosine": group_mass_cosine(
            group, mass, probability, log_mass
        ),
    }


def train_model(
    *,
    name: str,
    train_feature: np.ndarray,
    train_binary: np.ndarray,
    train_mass: np.ndarray,
    validation_feature: np.ndarray,
    validation_binary: np.ndarray,
    validation_mass: np.ndarray,
    validation_group: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[LifecycleNet, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LifecycleNet(train_feature.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    positive_count = max(int(np.sum(train_binary)), 1)
    negative_count = max(len(train_binary) - positive_count, 1)
    positive_weight = torch.tensor(
        min(negative_count / positive_count, 8.0), device=device
    )
    rng = np.random.default_rng(seed)
    best_state = None
    best_ap = -np.inf
    selected_epoch = 0
    epoch_rows = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_feature))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            feature = torch.from_numpy(train_feature[rows]).to(device)
            binary = torch.from_numpy(train_binary[rows].astype(np.float32)).to(device)
            mass = torch.from_numpy(train_mass[rows]).to(device)
            logit, predicted_log_mass = model(feature)
            existence_loss = nn.functional.binary_cross_entropy_with_logits(
                logit,
                binary,
                pos_weight=positive_weight,
                reduction="none",
            )
            sample_weight = 0.35 + torch.sqrt(torch.clamp(mass * 40.0, 0.0, 4.0))
            existence_loss = torch.mean(existence_loss * sample_weight)
            positive = binary > 0.5
            if torch.any(positive):
                mass_loss = nn.functional.smooth_l1_loss(
                    predicted_log_mass[positive],
                    torch.log(torch.clamp(mass[positive], min=1e-7)),
                )
            else:
                mass_loss = existence_loss * 0.0
            loss = existence_loss + 0.20 * mass_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability, log_mass = predict(model, validation_feature, device)
        validation = metrics(
            validation_binary,
            validation_mass,
            validation_group,
            probability,
            log_mass,
        )
        epoch_rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                **validation,
            }
        )
        if validation["existence_ap"] > best_ap:
            best_ap = validation["existence_ap"]
            selected_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"v375 {name} epoch={epoch} loss={np.mean(losses):.4f} "
                f"val_ap={validation['existence_ap']:.4f} "
                f"mass_cos={validation['group_mass_cosine']:.4f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("no synthetic lifecycle checkpoint selected")
    model.load_state_dict(best_state)
    return model, {
        "selected_epoch": selected_epoch,
        "best_validation_ap": float(best_ap),
        "epoch_rows": epoch_rows,
    }


def main() -> None:
    data = build_examples()
    feature = data["feature"].astype(np.float32)
    binary = data["binary"].astype(np.int8)
    mass = data["mass"].astype(np.float32)
    scene = data["scene"].astype(np.int64)
    group = data["group"].astype(np.int64)
    train_keep = np.isin(scene, TRAIN_SCENES)
    validation_keep = np.isin(scene, VALIDATION_SCENES)
    test_keep = np.isin(scene, TEST_SCENES)
    mean = feature[train_keep].mean(axis=0).astype(np.float32)
    std = np.maximum(feature[train_keep].std(axis=0), 1e-4).astype(np.float32)
    normalized = np.clip((feature - mean) / std, -8.0, 8.0).astype(np.float32)
    shuffled = shuffled_geometry(normalized, scene, 375_000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    train_reports = {}
    predictions = {}
    for mode_index, (name, mode_feature) in enumerate(
        (("full", normalized), ("shuffled_geometry", shuffled))
    ):
        model, train_report = train_model(
            name=name,
            train_feature=mode_feature[train_keep],
            train_binary=binary[train_keep],
            train_mass=mass[train_keep],
            validation_feature=mode_feature[validation_keep],
            validation_binary=binary[validation_keep],
            validation_mass=mass[validation_keep],
            validation_group=group[validation_keep],
            device=device,
            seed=3750 + mode_index,
        )
        probability, log_mass = predict(model, mode_feature[test_keep], device)
        models[name] = model
        train_reports[name] = train_report
        predictions[name] = (probability, log_mass)

    test_rows = {}
    for name, (probability, log_mass) in predictions.items():
        test_rows[name] = metrics(
            binary[test_keep],
            mass[test_keep],
            group[test_keep],
            probability,
            log_mass,
        )
        scene_rows = {}
        test_scene = scene[test_keep]
        for scene_id in TEST_SCENES:
            keep = test_scene == scene_id
            scene_rows[str(scene_id)] = metrics(
                binary[test_keep][keep],
                mass[test_keep][keep],
                group[test_keep][keep],
                probability[keep],
                log_mass[keep],
            )
        test_rows[name]["scenes"] = scene_rows

    ap_lift = float(
        test_rows["full"]["existence_ap"]
        - test_rows["shuffled_geometry"]["existence_ap"]
    )
    mass_lift = float(
        test_rows["full"]["group_mass_cosine"]
        - test_rows["shuffled_geometry"]["group_mass_cosine"]
    )
    scene_wins = int(
        sum(
            test_rows["full"]["scenes"][str(scene_id)]["existence_ap"]
            > test_rows["shuffled_geometry"]["scenes"][str(scene_id)]["existence_ap"]
            for scene_id in TEST_SCENES
        )
    )
    passed = bool(
        test_rows["full"]["existence_ap"] >= MIN_TEST_AP
        and ap_lift >= MIN_AP_CONTROL_LIFT
        and mass_lift >= MIN_MASS_CONTROL_LIFT
        and scene_wins >= MIN_SCENE_WINS
    )
    torch.save(
        {
            "state_dict": models["full"].state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_dimensions": lgf.feature_dimensions(),
            "selected_epoch": train_reports["full"]["selected_epoch"],
        },
        CHECKPOINT,
    )
    summary = {
        "example_count": int(len(feature)),
        "train_count": int(np.sum(train_keep)),
        "validation_count": int(np.sum(validation_keep)),
        "test_count": int(np.sum(test_keep)),
        "full": test_rows["full"],
        "shuffled_geometry": test_rows["shuffled_geometry"],
        "existence_ap_control_lift": ap_lift,
        "group_mass_cosine_control_lift": mass_lift,
        "test_scene_wins": scene_wins,
    }
    report = {
        "method": VERSION,
        "status": "synthetic_pretraining_passed" if passed else "analysis_only_no_transfer",
        "scope": "unseen procedural-scene lifecycle pretraining; no official H/test target/ZIP",
        "primary_hypothesis": (
            "Point-cloud-compatible ray and support features contain universal path "
            "birth/death and conditional-energy information across independent Sionna scenes."
        ),
        "compliance": {
            "uses": [
                "v374 independent procedural Sionna scenes",
                "occupancy/ray features reproducible from an official point cloud",
            ],
            "does_not_use": [
                "restored Etoile scene, official positions/H, synthetic official-test targets",
                "v212 teacher, hidden/local/online truth, or submission outputs",
            ],
        },
        "parameters": {
            "train_scenes": list(TRAIN_SCENES),
            "validation_scenes": list(VALIDATION_SCENES),
            "test_scenes": list(TEST_SCENES),
            "families_per_scene": FAMILIES_PER_SCENE,
            "feature_dimensions": lgf.feature_dimensions(),
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "control": "point/ray geometry shuffled within each scene",
        },
        "training": train_reports,
        "summary": summary,
        "decision_rule": {
            "test_existence_ap": f">={MIN_TEST_AP}",
            "existence_ap_control_lift": f">={MIN_AP_CONTROL_LIFT}",
            "group_mass_cosine_control_lift": f">={MIN_MASS_CONTROL_LIFT}",
            "test_scene_wins": f">={MIN_SCENE_WINS}/3",
            "passed": passed,
        },
        "outputs": {
            "examples": PATHS.relative(EXAMPLES),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "report": PATHS.relative(REPORT),
            "official_prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_rule"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
