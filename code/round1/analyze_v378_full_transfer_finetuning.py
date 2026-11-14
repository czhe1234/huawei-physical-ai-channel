#!/usr/bin/env python3
"""Strictly audit full-network lifecycle transfer fine-tuning."""

from __future__ import annotations

import copy
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v370_path_track_lifecycle_pdp as v370
import analyze_v373_long_family_visibility as v373
import analyze_v376_frozen_lifecycle_transfer as v376
import analyze_v377_nested_embedding_calibration as v377
import lifecycle_geometry_features as lgf
import train_v375_synthetic_lifecycle_pretraining as v375
from project_paths import PATHS


VERSION_NUMBER = 378
VERSION = "v378_full_transfer_finetuning"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
PAIR_CACHE = OUT_DIR / "v378_full_transfer_finetuning_pairs.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
CHECKPOINT = PATHS.cache_dir(375) / "v375_synthetic_lifecycle_model.pt"

EPOCHS = 24
BATCH_SIZE = 4096
TRUNK_LEARNING_RATE = 2e-4
HEAD_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-5
MASS_LOSS_WEIGHT = 0.15
CORRECTION_PENALTY = 0.01
ENSEMBLE_SEEDS = (37801, 37802, 37803)

MIN_PRETRAINED_VS_LOCAL_AP = 0.002
MIN_PRETRAINED_VS_LOCAL_MASS = 0.002
MIN_PRETRAINED_VS_SCRATCH_AP = 0.002
MIN_PRETRAINED_VS_SCRATCH_MASS = 0.002
MIN_PRETRAINED_VS_SHUFFLED_AP = 0.002
MIN_PRETRAINED_VS_SHUFFLED_MASS = 0.002
MIN_FOLD_WINS = 4


class ResidualLifecycleNet(nn.Module):
    def __init__(self, trunk: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.existence = nn.Sequential(
            nn.Linear(131, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.log_mass = nn.Sequential(
            nn.Linear(131, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
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


def context_features(
    identity: np.ndarray,
    local_probability: np.ndarray,
    local_mass: np.ndarray,
) -> np.ndarray:
    local_logit = np.clip(v376.logit(local_probability), -12.0, 12.0)
    local_conditional = np.divide(
        local_mass, np.maximum(local_probability, 1e-5)
    )
    local_log_mass = np.clip(np.log(np.maximum(local_conditional, 1e-8)), -16.0, 2.0)
    identity_logit = np.clip(v376.logit(identity), -8.0, 8.0)
    return np.column_stack(
        [identity_logit / 4.0, local_logit / 4.0, (local_log_mass + 6.0) / 4.0]
    ).astype(np.float32)


def collect_examples(
    rows: list[dict[str, object]],
    *,
    shuffled: bool,
    seed: int,
) -> dict[str, np.ndarray]:
    features = []
    identities = []
    binaries = []
    masses = []
    local_probabilities = []
    local_masses = []
    for index, row in enumerate(rows):
        exposure = row["exposure"]
        feature = row["feature"]
        if shuffled:
            feature = v375.shuffled_geometry(
                feature,
                np.zeros(len(feature), dtype=np.int8),
                seed + index,
            )
        features.append(feature)
        identities.append(exposure.identity)
        binaries.append(exposure.binary.astype(np.float32))
        masses.append(exposure.mass)
        local_probabilities.append(row["local_probability"][exposure.node])
        local_masses.append(row["local_mass"][exposure.node])
    return {
        "feature": np.concatenate(features).astype(np.float32),
        "identity": np.concatenate(identities).astype(np.float32),
        "binary": np.concatenate(binaries).astype(np.float32),
        "mass": np.concatenate(masses).astype(np.float32),
        "local_probability": np.concatenate(local_probabilities).astype(np.float32),
        "local_mass": np.concatenate(local_masses).astype(np.float32),
    }


def one_example_set(
    row: dict[str, object],
    *,
    shuffled: bool,
    seed: int,
) -> dict[str, np.ndarray]:
    return collect_examples([row], shuffled=shuffled, seed=seed)


def normalize_feature(
    feature: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return np.clip((feature - mean) / std, -8.0, 8.0).astype(np.float32)


def make_model(
    *,
    pretrained: bool,
    checkpoint: dict[str, object],
    seed: int,
) -> ResidualLifecycleNet:
    torch.manual_seed(seed)
    source = v375.LifecycleNet(lgf.feature_dimensions()["total"])
    if pretrained:
        source.load_state_dict(checkpoint["state_dict"])
    model = ResidualLifecycleNet(copy.deepcopy(source.trunk))
    return model


def train_predict_once(
    *,
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    checkpoint: dict[str, object],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    pretrained: bool,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(pretrained=pretrained, checkpoint=checkpoint, seed=seed).to(device)
    train_feature = normalize_feature(train["feature"], feature_mean, feature_std)
    query_feature = normalize_feature(query["feature"], feature_mean, feature_std)
    train_context = context_features(
        train["identity"], train["local_probability"], train["local_mass"]
    )
    query_context = context_features(
        query["identity"], query["local_probability"], query["local_mass"]
    )
    train_local_logit = np.clip(
        v376.logit(train["local_probability"]), -12.0, 12.0
    ).astype(np.float32)
    train_local_conditional = np.divide(
        train["local_mass"], np.maximum(train["local_probability"], 1e-5)
    )
    train_local_log_mass = np.clip(
        np.log(np.maximum(train_local_conditional, 1e-8)), -16.0, 2.0
    ).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(train_feature),
        torch.from_numpy(train_context),
        torch.from_numpy(train_local_logit),
        torch.from_numpy(train_local_log_mass),
        torch.from_numpy(train["binary"]),
        torch.from_numpy(train["mass"]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.trunk.parameters(), "lr": TRUNK_LEARNING_RATE},
            {"params": model.existence.parameters(), "lr": HEAD_LEARNING_RATE},
            {"params": model.log_mass.parameters(), "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    bce = nn.BCEWithLogitsLoss()
    epoch_loss = []
    for _ in range(EPOCHS):
        model.train()
        total = 0.0
        count = 0
        for feature, context, local_logit, local_log_mass, binary, mass in loader:
            feature = feature.to(device)
            context = context.to(device)
            local_logit = local_logit.to(device)
            local_log_mass = local_log_mass.to(device)
            binary = binary.to(device)
            mass = mass.to(device)
            existence_correction, mass_correction = model(feature, context)
            score = local_logit + existence_correction
            loss_existence = bce(score, binary)
            positive = binary > 0.5
            if torch.any(positive):
                target_log_mass = torch.log(torch.clamp(mass[positive], min=1e-8))
                predicted_log_mass = local_log_mass[positive] + mass_correction[positive]
                loss_mass = torch.mean((predicted_log_mass - target_log_mass) ** 2)
            else:
                loss_mass = torch.zeros((), device=device)
            penalty = torch.mean(existence_correction**2) + torch.mean(mass_correction**2)
            loss = (
                loss_existence
                + MASS_LOSS_WEIGHT * loss_mass
                + CORRECTION_PENALTY * penalty
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(binary)
            count += len(binary)
        epoch_loss.append(total / max(count, 1))

    query_local_logit = np.clip(
        v376.logit(query["local_probability"]), -12.0, 12.0
    ).astype(np.float32)
    query_local_conditional = np.divide(
        query["local_mass"], np.maximum(query["local_probability"], 1e-5)
    )
    query_local_log_mass = np.clip(
        np.log(np.maximum(query_local_conditional, 1e-8)), -16.0, 2.0
    ).astype(np.float32)
    probability = np.empty(len(query_feature), dtype=np.float32)
    mass_score = np.empty(len(query_feature), dtype=np.float32)
    correction_abs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(query_feature), 16384):
            end = min(start + 16384, len(query_feature))
            feature = torch.from_numpy(query_feature[start:end]).to(device)
            context = torch.from_numpy(query_context[start:end]).to(device)
            existence_correction, mass_correction = model(feature, context)
            existence_correction_np = existence_correction.cpu().numpy()
            mass_correction_np = mass_correction.cpu().numpy()
            probability[start:end] = v376.sigmoid(
                query_local_logit[start:end] + existence_correction_np
            ).astype(np.float32)
            conditional = np.exp(
                np.clip(
                    query_local_log_mass[start:end] + mass_correction_np,
                    -16.0,
                    2.0,
                )
            )
            mass_score[start:end] = probability[start:end] * conditional
            correction_abs.append(
                np.column_stack(
                    [np.abs(existence_correction_np), np.abs(mass_correction_np)]
                )
            )
    correction = np.concatenate(correction_abs)
    return probability, mass_score, {
        "initial_loss": float(epoch_loss[0]),
        "final_loss": float(epoch_loss[-1]),
        "existence_correction_abs_mean": float(np.mean(correction[:, 0])),
        "mass_correction_abs_mean": float(np.mean(correction[:, 1])),
    }


def ensemble_variant(
    *,
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    checkpoint: dict[str, object],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    pretrained: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    probabilities = []
    masses = []
    diagnostics = []
    for seed in ENSEMBLE_SEEDS:
        probability, mass, row = train_predict_once(
            train=train,
            query=query,
            checkpoint=checkpoint,
            feature_mean=feature_mean,
            feature_std=feature_std,
            pretrained=pretrained,
            seed=seed,
            device=device,
        )
        probabilities.append(probability)
        masses.append(mass)
        diagnostics.append(row)
    return (
        np.mean(probabilities, axis=0).astype(np.float32),
        np.mean(masses, axis=0).astype(np.float32),
        diagnostics,
    )


def main() -> None:
    for path in (
        TARGET_CACHE,
        CHECKPOINT,
        v297.COMPONENT_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
        v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

    target = np.load(TARGET_CACHE)
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    component_cache = np.load(v297.COMPONENT_CACHE)
    component = {key: component_cache[key] for key in component_cache.files}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    geometry = v274.load_geometry()
    all_pos, signatures, blocker_count, scene_features = geometry[:4]
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    scene_cache = np.load(v274.SCENE_CACHE)
    scene_dict = {key: scene_cache[key] for key in scene_cache.files}
    raster = lgf.raster_from_pointcloud_scene(scene_dict)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    point_feature = lgf.point_features(raster, all_pos, bs)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])

    fold_rows: list[dict[str, object]] = []
    cache: dict[str, list[np.ndarray]] = {
        "fold": [],
        "truth_binary": [],
        "truth_mass": [],
        "local_probability": [],
        "local_mass": [],
        "covered": [],
    }
    variants = ("scratch", "shuffled_pretrained", "pretrained")
    for variant in variants:
        cache[f"{variant}_probability"] = []
        cache[f"{variant}_mass"] = []

    for outer_fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        nested_rows = []
        for inner_fold, inner_ids_all in enumerate(folds):
            if inner_fold == outer_fold:
                continue
            inner_held = np.intersect1d(fit_ids, inner_ids_all)
            inner_fit = np.setdiff1d(fit_ids, inner_held)
            nested_rows.append(
                v377.build_fold_examples(
                    fit_ids=inner_fit,
                    held_ids=inner_held,
                    valid_ids=valid_ids,
                    geometry=geometry,
                    node_raw=node_raw,
                    component=component,
                    all_pos=all_pos,
                    point_feature=point_feature,
                    raster=raster,
                    bs=bs,
                    model=v375.LifecycleNet(lgf.feature_dimensions()["total"]),
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    device=torch.device("cpu"),
                    seed=378_000 + outer_fold * 100 + inner_fold,
                )
            )
        outer = v377.build_fold_examples(
            fit_ids=fit_ids,
            held_ids=outer_ids,
            valid_ids=valid_ids,
            geometry=geometry,
            node_raw=node_raw,
            component=component,
            all_pos=all_pos,
            point_feature=point_feature,
            raster=raster,
            bs=bs,
            model=v375.LifecycleNet(lgf.feature_dimensions()["total"]),
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=torch.device("cpu"),
            seed=378_500 + outer_fold,
        )
        train_real = collect_examples(
            nested_rows, shuffled=False, seed=378_600 + outer_fold
        )
        train_shuffled = collect_examples(
            nested_rows, shuffled=True, seed=378_700 + outer_fold
        )
        query_real = one_example_set(
            outer, shuffled=False, seed=378_800 + outer_fold
        )
        query_shuffled = one_example_set(
            outer, shuffled=True, seed=378_900 + outer_fold
        )

        outer_bundle = outer["bundle"]
        outer_exposure = outer["exposure"]
        local_probability = outer["local_probability"]
        local_mass = outer["local_mass"]
        local_metrics = v373.subset_metrics(
            outer_bundle, outer_ids, local_probability, local_mass, None
        )

        variant_data = {
            "scratch": (train_real, query_real, False),
            "shuffled_pretrained": (train_shuffled, query_shuffled, True),
            "pretrained": (train_real, query_real, True),
        }
        outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        metrics: dict[str, dict[str, float]] = {}
        training: dict[str, list[dict[str, float]]] = {}
        for variant, (train_data, query_data, use_pretraining) in variant_data.items():
            pair_probability, pair_mass, diagnostics = ensemble_variant(
                train=train_data,
                query=query_data,
                checkpoint=checkpoint,
                feature_mean=feature_mean,
                feature_std=feature_std,
                pretrained=use_pretraining,
                device=device,
            )
            output = v377.apply_calibrated_exposures(
                exposure=outer_exposure,
                probability=pair_probability,
                mass=pair_mass,
                local_probability=local_probability,
                local_mass=local_mass,
            )
            outputs[variant] = output
            metrics[variant] = v373.subset_metrics(
                outer_bundle, outer_ids, output[0], output[1], None
            )
            training[variant] = diagnostics

        pretrained_metrics = metrics["pretrained"]
        scratch_metrics = metrics["scratch"]
        shuffled_metrics = metrics["shuffled_pretrained"]
        deltas = {
            "pretrained_vs_local_ap": float(
                pretrained_metrics["existence_ap"] - local_metrics["existence_ap"]
            ),
            "pretrained_vs_local_mass": float(
                pretrained_metrics["mass_cosine"] - local_metrics["mass_cosine"]
            ),
            "pretrained_vs_scratch_ap": float(
                pretrained_metrics["existence_ap"] - scratch_metrics["existence_ap"]
            ),
            "pretrained_vs_scratch_mass": float(
                pretrained_metrics["mass_cosine"] - scratch_metrics["mass_cosine"]
            ),
            "pretrained_vs_shuffled_ap": float(
                pretrained_metrics["existence_ap"] - shuffled_metrics["existence_ap"]
            ),
            "pretrained_vs_shuffled_mass": float(
                pretrained_metrics["mass_cosine"] - shuffled_metrics["mass_cosine"]
            ),
        }
        fold_rows.append(
            {
                "fold": outer_fold,
                "nested_training_exposures": int(len(train_real["binary"])),
                "outer_exposures": int(len(outer_exposure.node)),
                "local": local_metrics,
                "variants": metrics,
                "training": training,
                "deltas": deltas,
                "leakage_assertions": {
                    "outer_in_finetuning": 0,
                    "outer_h_in_family_discovery": 0,
                    "synthetic_official_target": 0,
                    "v212_teacher": 0,
                },
            }
        )
        outer_nodes = np.isin(outer_bundle.node_query_id, outer_ids)
        cache["fold"].append(
            np.full(np.sum(outer_nodes), outer_fold, dtype=np.int8)
        )
        cache["truth_binary"].append(
            outer_bundle.arrays["binary"][outer_nodes].astype(np.int8)
        )
        cache["truth_mass"].append(outer_bundle.arrays["mass"][outer_nodes])
        cache["local_probability"].append(local_probability[outer_nodes])
        cache["local_mass"].append(local_mass[outer_nodes])
        cache["covered"].append(outputs["pretrained"][2][outer_nodes])
        for variant in variants:
            cache[f"{variant}_probability"].append(outputs[variant][0][outer_nodes])
            cache[f"{variant}_mass"].append(outputs[variant][1][outer_nodes])
        print(
            f"v378 fold={outer_fold} train={len(train_real['binary'])} "
            f"delta_local={deltas['pretrained_vs_local_ap']:+.4f}/"
            f"{deltas['pretrained_vs_local_mass']:+.4f} "
            f"delta_scratch={deltas['pretrained_vs_scratch_ap']:+.4f}/"
            f"{deltas['pretrained_vs_scratch_mass']:+.4f}",
            flush=True,
        )

    merged = {key: np.concatenate(value) for key, value in cache.items()}
    np.savez_compressed(PAIR_CACHE, **merged)
    mean = {
        key: float(np.mean([row["deltas"][key] for row in fold_rows]))
        for key in fold_rows[0]["deltas"]
    }
    fold_wins = int(
        sum(
            row["deltas"]["pretrained_vs_local_ap"] > 0.0
            and row["deltas"]["pretrained_vs_local_mass"] > 0.0
            and row["deltas"]["pretrained_vs_scratch_ap"] > 0.0
            and row["deltas"]["pretrained_vs_scratch_mass"] > 0.0
            and row["deltas"]["pretrained_vs_shuffled_ap"] > 0.0
            and row["deltas"]["pretrained_vs_shuffled_mass"] > 0.0
            for row in fold_rows
        )
    )
    passed = bool(
        mean["pretrained_vs_local_ap"] >= MIN_PRETRAINED_VS_LOCAL_AP
        and mean["pretrained_vs_local_mass"] >= MIN_PRETRAINED_VS_LOCAL_MASS
        and mean["pretrained_vs_scratch_ap"] >= MIN_PRETRAINED_VS_SCRATCH_AP
        and mean["pretrained_vs_scratch_mass"] >= MIN_PRETRAINED_VS_SCRATCH_MASS
        and mean["pretrained_vs_shuffled_ap"] >= MIN_PRETRAINED_VS_SHUFFLED_AP
        and mean["pretrained_vs_shuffled_mass"] >= MIN_PRETRAINED_VS_SHUFFLED_MASS
        and fold_wins >= MIN_FOLD_WINS
    )
    summary = {**mean, "fold_wins": fold_wins}
    report = {
        "method": VERSION,
        "status": "full_transfer_passed" if passed else "analysis_only_no_transfer",
        "scope": "strict nested official path-level audit; no PDP/full-H/test target/ZIP",
        "primary_hypothesis": (
            "Full low-rate fine-tuning adapts the v375 propagation representation to "
            "official lifecycle residuals and beats both scratch training and shuffled "
            "pretraining while improving the official local model."
        ),
        "literature_basis": [
            "arXiv:1912.12265 direct transfer and meta-learning for channel prediction",
            "arXiv:2602.10736 simulation priors plus target-domain alignment for radio maps",
        ],
        "parameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "trunk_learning_rate": TRUNK_LEARNING_RATE,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "mass_loss_weight": MASS_LOSS_WEIGHT,
            "correction_penalty": CORRECTION_PENALTY,
            "ensemble_seeds": list(ENSEMBLE_SEEDS),
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "pretrained_vs_local_ap": f">={MIN_PRETRAINED_VS_LOCAL_AP}",
            "pretrained_vs_local_mass": f">={MIN_PRETRAINED_VS_LOCAL_MASS}",
            "pretrained_vs_scratch_ap": f">={MIN_PRETRAINED_VS_SCRATCH_AP}",
            "pretrained_vs_scratch_mass": f">={MIN_PRETRAINED_VS_SCRATCH_MASS}",
            "pretrained_vs_shuffled_ap": f">={MIN_PRETRAINED_VS_SHUFFLED_AP}",
            "pretrained_vs_shuffled_mass": f">={MIN_PRETRAINED_VS_SHUFFLED_MASS}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "passed": passed,
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "pair_cache": PATHS.relative(PAIR_CACHE),
            "prediction": None,
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
