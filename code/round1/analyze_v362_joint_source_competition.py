#!/usr/bin/env python3
"""Strict KNN-anchored joint virtual-source competition model."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v311_effective_source_existence_boundary as v311
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v354_nested_virtual_source_transport as v354
import analyze_v355_virtual_source_lifecycle_model as v355
import analyze_v359_fingerprint_subfamily_identifiability as v359
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 362
VERSION = "v362_joint_source_competition"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v362_joint_source_competition_oof.npz"

HIDDEN_DIM = 96
HEADS = 4
LAYERS = 2
EPOCHS = 100
BATCH_SIZE = 64
MAX_COMPOSITION_MIX = 0.35
MAX_TOTAL_LOGIT_DELTA = 0.75
MIN_AP_LIFT = 0.05
MIN_MASS_COSINE_LIFT = 0.03
MIN_RENDERER_DELTA = 0.003
MIN_FOLD_WINS = 4


class SourceCompetitionModel(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_dim + 4, HIDDEN_DIM),
            nn.SiLU(),
            nn.LayerNorm(HIDDEN_DIM),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=HEADS,
            dim_feedforward=HIDDEN_DIM * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.source_score = nn.Linear(HIDDEN_DIM, 1)
        self.mix_gate = nn.Linear(HIDDEN_DIM, 1)
        self.total_delta = nn.Linear(HIDDEN_DIM, 1)

    def forward(
        self, feature: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total = torch.clamp(base.sum(dim=1, keepdim=True), 1e-5, 1.0 - 1e-5)
        composition = base / torch.clamp(total, min=1e-8)
        base_channels = torch.stack(
            [
                torch.sqrt(torch.clamp(base, min=0.0)),
                torch.log1p(100.0 * torch.clamp(base, min=0.0)) / 5.0,
                composition,
                total.expand_as(base),
            ],
            dim=2,
        )
        hidden = self.encoder(self.input(torch.cat([feature, base_channels], dim=2)))
        learned_composition = torch.softmax(self.source_score(hidden).squeeze(2), dim=1)
        pooled = hidden.mean(dim=1)
        mix = MAX_COMPOSITION_MIX * torch.sigmoid(self.mix_gate(pooled))
        predicted_composition = (
            (1.0 - mix) * composition + mix * learned_composition
        )
        base_logit = torch.log(total / (1.0 - total))
        delta = MAX_TOTAL_LOGIT_DELTA * torch.tanh(self.total_delta(pooled))
        predicted_total = torch.sigmoid(base_logit + delta)
        return predicted_total * predicted_composition, mix.squeeze(1), delta.squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def crossfit_training_set(
    fit_ids: np.ndarray,
    fit_target: np.ndarray,
    centers: np.ndarray,
    train_pos: np.ndarray,
    scene_feature: np.ndarray,
    signature: np.ndarray,
    scene: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_lookup = np.full(len(train_pos), -1, dtype=np.int64)
    row_lookup[fit_ids] = np.arange(len(fit_ids), dtype=np.int64)
    features: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for query_ids in v315.disjoint_folds(fit_ids, train_pos[:, :2]):
        bank_ids = np.setdiff1d(fit_ids, query_ids)
        bank_rows = row_lookup[bank_ids]
        query_rows = row_lookup[query_ids]
        if np.any(bank_rows < 0) or np.any(query_rows < 0):
            raise AssertionError("v362 crossfit lookup failed")
        bank_target = fit_target[bank_rows]
        features.append(
            v355.pair_features(
                query_ids,
                bank_ids,
                bank_target,
                centers,
                train_pos,
                scene_feature,
                signature,
                scene,
            )
        )
        bases.append(
            v311.knn_prediction(
                query_ids, bank_ids, bank_target, train_pos, k=v354.K_ANCHOR
            )
        )
        targets.append(fit_target[query_rows])
    return np.concatenate(features), np.concatenate(bases), np.concatenate(targets)


def standardize(
    train_feature: np.ndarray, outer_feature: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean = train_feature.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train_feature.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return (
        ((train_feature - mean) / std).astype(np.float32),
        ((outer_feature - mean) / std).astype(np.float32),
        {"mean": mean, "std": std},
    )


def source_loss(
    prediction: torch.Tensor, target: torch.Tensor, mix: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction_total = prediction.sum(dim=1)
    target_total = target.sum(dim=1)
    positive = target_total > 1e-6
    pointwise = F.smooth_l1_loss(prediction, target, beta=0.02)
    total_loss = F.smooth_l1_loss(prediction_total, target_total, beta=0.03)
    if torch.any(positive):
        cosine = F.cosine_similarity(
            prediction[positive], target[positive], dim=1, eps=1e-8
        )
        composition_loss = 1.0 - cosine.mean()
    else:
        composition_loss = pointwise * 0.0
    regularization = 0.01 * mix.mean()
    loss = 5.0 * pointwise + 2.0 * total_loss + composition_loss + regularization
    return loss, {
        "pointwise": float(pointwise.detach().cpu()),
        "total": float(total_loss.detach().cpu()),
        "composition": float(composition_loss.detach().cpu()),
        "mix_regularization": float(regularization.detach().cpu()),
    }


def train_model(
    feature: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[SourceCompetitionModel, list[dict[str, float]]]:
    set_seed(seed)
    model = SourceCompetitionModel(feature.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-4)
    feature_tensor = torch.from_numpy(feature)
    base_tensor = torch.from_numpy(base)
    target_tensor = torch.from_numpy(target)
    rng = np.random.default_rng(seed + 17)
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        order = rng.permutation(len(feature))
        for start in range(0, len(feature), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            ids_tensor = torch.from_numpy(ids)
            x = feature_tensor[ids_tensor].to(device)
            b = base_tensor[ids_tensor].to(device)
            y = target_tensor[ids_tensor].to(device)
            prediction, mix, _delta = model(x, b)
            loss, _parts = source_loss(prediction, y, mix)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                sample = slice(0, min(256, len(feature)))
                prediction, mix, delta = model(
                    feature_tensor[sample].to(device), base_tensor[sample].to(device)
                )
                metric = v311.evaluate_prediction(
                    prediction.cpu().numpy(), target[sample]
                )
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "sample_pair_ap": metric["pair_ap"],
                    "sample_mass_cosine": metric["mass_cosine"],
                    "sample_mix_mean": float(mix.mean().cpu()),
                    "sample_total_delta_mean": float(delta.mean().cpu()),
                }
            )
    return model, history


@torch.no_grad()
def predict(
    model: SourceCompetitionModel,
    feature: np.ndarray,
    base: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty_like(base, dtype=np.float32)
    mix = np.empty(len(base), dtype=np.float32)
    delta = np.empty(len(base), dtype=np.float32)
    for start in range(0, len(base), 256):
        end = min(start + 256, len(base))
        value, value_mix, value_delta = model(
            torch.from_numpy(feature[start:end]).to(device),
            torch.from_numpy(base[start:end]).to(device),
        )
        output[start:end] = value.cpu().numpy()
        mix[start:end] = value_mix.cpu().numpy()
        delta[start:end] = value_delta.cpu().numpy()
    return output, mix, delta


def metric_delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: float(left[key] - right[key]) for key in ("sh", "d", "weighted")}


def main() -> None:
    for path in (
        v354.TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v354.TRAIN_CANDIDATES,
        v273.SCENE_CACHE,
        v273.FEATURE_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    target = np.load(v354.TARGET_CACHE)
    truth_sh = v354.normalize(target["sh"].astype(np.float32))
    truth_d = v354.normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    train_xy = train_pos[:, :2]
    observation, observation_report = v354.build_physical_observations()
    scene_data = np.load(v273.SCENE_CACHE)
    scene = {key: scene_data[key] for key in scene_data.files}
    scene_feature = scene["train_features"].astype(np.float32)
    signature = np.load(v273.FEATURE_CACHE)["train_signature"].astype(np.float32)
    folds = v315.disjoint_folds(valid_ids, train_xy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_rows: list[dict[str, object]] = []
    oof_payload: dict[str, np.ndarray] = {}
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        centers, source_report = v311.discover_sources(observation, fit_ids)
        fit = v359.assign_observations(observation, fit_ids, centers, train_pos)
        outer = v359.assign_observations(observation, outer_ids, centers, train_pos)
        train_feature, train_base, train_target = crossfit_training_set(
            fit_ids,
            fit.target,
            centers,
            train_pos,
            scene_feature,
            signature,
            scene,
        )
        outer_feature = v355.pair_features(
            outer_ids,
            fit_ids,
            fit.target,
            centers,
            train_pos,
            scene_feature,
            signature,
            scene,
        )
        outer_base = v311.knn_prediction(
            outer_ids, fit_ids, fit.target, train_pos, k=v354.K_ANCHOR
        )
        train_feature, outer_feature, scaling = standardize(
            train_feature, outer_feature
        )
        model, history = train_model(
            train_feature,
            train_base,
            train_target,
            VERSION_NUMBER * 10 + fold,
            device,
        )
        predicted_mass, mix, total_delta = predict(
            model, outer_feature, outer_base, device
        )
        source_metrics = {
            "knn": v311.evaluate_prediction(outer_base, outer.target),
            "joint_set": v311.evaluate_prediction(predicted_mass, outer.target),
        }
        ap_lift = source_metrics["joint_set"]["pair_ap"] - source_metrics["knn"]["pair_ap"]
        mass_lift = (
            source_metrics["joint_set"]["mass_cosine"]
            - source_metrics["knn"]["mass_cosine"]
        )
        neighbor = v354.neighbors(outer_ids, fit_ids, train_xy, False)
        if np.intersect1d(np.unique(neighbor.ids), outer_ids).size:
            raise AssertionError("outer H entered v362 KNN current")
        anchor = v354.anchor_from_neighbors(truth_sh, neighbor)
        knn_prediction, knn_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit.target,
            centers,
            train_xy,
        )
        model_prediction, model_diagnostics = v354.transport(
            outer_ids,
            fit_ids,
            neighbor,
            anchor,
            fit.target,
            centers,
            train_xy,
            query_mass_override=predicted_mass,
        )
        renderer_metrics = {
            "knn": v354.metrics(
                knn_prediction, truth_sh[outer_ids], truth_d[outer_ids]
            ),
            "joint_set": v354.metrics(
                model_prediction, truth_sh[outer_ids], truth_d[outer_ids]
            ),
        }
        renderer_delta = metric_delta(
            renderer_metrics["joint_set"], renderer_metrics["knn"]
        )
        passed_fold = bool(
            ap_lift >= MIN_AP_LIFT
            and mass_lift >= MIN_MASS_COSINE_LIFT
            and renderer_delta["weighted"] >= MIN_RENDERER_DELTA
        )
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "source_discovery": source_report,
                "fit_assignment": fit.report,
                "outer_assignment": outer.report,
                "training_history": history,
                "feature_scaling": {
                    "dimension": int(len(scaling["mean"])),
                    "mean_abs": float(np.mean(np.abs(scaling["mean"]))),
                    "std_mean": float(np.mean(scaling["std"])),
                },
                "source_metrics": source_metrics,
                "source_lift": {
                    "pair_ap": float(ap_lift),
                    "mass_cosine": float(mass_lift),
                },
                "renderer_metrics": renderer_metrics,
                "renderer_delta": renderer_delta,
                "mix": summarize(mix),
                "total_logit_delta": summarize(total_delta),
                "predicted_total_mass": summarize(predicted_mass.sum(axis=1)),
                "true_total_mass": summarize(outer.target.sum(axis=1)),
                "passed_fold": passed_fold,
                "diagnostics": {
                    "knn": {
                        key: summarize(value) for key, value in knn_diagnostics.items()
                    },
                    "joint_set": {
                        key: summarize(value) for key, value in model_diagnostics.items()
                    },
                },
                "leakage_assertions": {
                    "outer_in_source_discovery": 0,
                    "outer_in_training_support_or_knn": 0,
                    "training_query_in_own_crossfit_support": 0,
                },
            }
        )
        oof_payload[f"fold_{fold}_ids"] = outer_ids
        oof_payload[f"fold_{fold}_anchor"] = anchor
        oof_payload[f"fold_{fold}_knn"] = knn_prediction
        oof_payload[f"fold_{fold}_joint_set"] = model_prediction
        print(
            f"v362 fold={fold} AP={ap_lift:+.4f} mass={mass_lift:+.4f} "
            f"renderer={renderer_delta['weighted']:+.6f} mix={mix.mean():.3f} "
            f"pass={passed_fold}",
            flush=True,
        )

    mean_source_lift = {
        key: float(np.mean([row["source_lift"][key] for row in fold_rows]))
        for key in ("pair_ap", "mass_cosine")
    }
    mean_renderer_delta = {
        key: float(np.mean([row["renderer_delta"][key] for row in fold_rows]))
        for key in ("sh", "d", "weighted")
    }
    fold_wins = int(sum(row["passed_fold"] for row in fold_rows))
    passed = bool(
        mean_source_lift["pair_ap"] >= MIN_AP_LIFT
        and mean_source_lift["mass_cosine"] >= MIN_MASS_COSINE_LIFT
        and mean_renderer_delta["weighted"] >= MIN_RENDERER_DELTA
        and fold_wins >= MIN_FOLD_WINS
    )
    np.savez_compressed(OOF_OUT, **oof_payload)
    report = {
        "method": VERSION,
        "status": "strict_joint_source_passed" if passed else "analysis_only_no_candidate",
        "round": 4,
        "primary_hypothesis": (
            "Source masses compete within each normalized multipath mixture; a KNN-anchored set "
            "model can learn bounded joint reallocation and total-mass correction that independent "
            "per-source lifecycle regressors cannot represent."
        ),
        "compliance": {
            "uses": [
                "official outer-fit H hard source targets",
                "official train coordinates",
                "official Round1_Map point-cloud route features",
            ],
            "does_not_use": [
                "outer H in source discovery, model features, KNN, or training",
                "v207/v212/synthetic truth or teacher",
                "restored RT channel/material coefficients",
                "local or online labels",
            ],
        },
        "parameters": {
            "model": "2-layer 96-dim 4-head KNN-anchored source Set Transformer",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "maximum_composition_mix": MAX_COMPOSITION_MIX,
            "maximum_total_logit_delta": MAX_TOTAL_LOGIT_DELTA,
            "training_features": "five-fold crossfit v355 physical/support/route features",
            "parameter_search": "none",
            "minimum_pair_ap_lift": MIN_AP_LIFT,
            "minimum_mass_cosine_lift": MIN_MASS_COSINE_LIFT,
            "minimum_renderer_delta": MIN_RENDERER_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
        },
        "observation": observation_report,
        "fold_rows": fold_rows,
        "decision": {
            "mean_source_lift": mean_source_lift,
            "mean_renderer_delta": mean_renderer_delta,
            "fold_wins": fold_wins,
            "passed": passed,
            "next_action": (
                "evaluate the joint source model in the strict final-current chain"
                if passed
                else "stop replacing lifecycle regressors and seek a new observable source-state variable"
            ),
        },
        "outputs": {
            "oof": PATHS.relative(OOF_OUT),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
