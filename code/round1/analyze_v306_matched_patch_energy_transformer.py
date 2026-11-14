#!/usr/bin/env python3
"""Predict official-pointcloud path energy from v305 identity-matched labels."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v254_route_topology_spectral_field as v254
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import build_official_pointcloud_scene as pointcloud_scene
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 306
VERSION = "v306_matched_patch_energy_transformer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
TEST_CANDIDATES = OUT_DIR / "v306_test_wall_reflection_candidates.npz"
LABEL_CACHE = OUT_DIR / "v306_train_matched_path_labels.npz"
TARGETS_OUT = OUT_DIR / "v306_matched_patch_delay_targets.npz"
OUTPUT = OUT_DIR / "v306_Round1_Test_Channel.npy"

TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
TARGET_CACHE = v297.TARGET_CACHE
BASE = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"
SCENE_CACHE = v264.SCENE_CACHE

REFLECTIONS = 128
PATH_COUNT = REFLECTIONS + 1
OUTER_SEEDS = (306, 1306, 2306, 3306, 4306)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 80
EVAL_EVERY = 5
BATCH_SIZE = 64
K_ANCHOR = 16
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)
MIN_OFFICIAL_PDP_DELTA = 0.01
FINAL_SEEDS = (30601, 30631, 30661)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_test_candidates(test_pos: np.ndarray, patches: np.ndarray) -> dict[str, np.ndarray]:
    if TEST_CANDIDATES.is_file():
        data = np.load(TEST_CANDIDATES)
        return {key: data[key] for key in data.files}
    built = pointcloud_scene.build_reflection_candidates(
        test_pos, patches, top_k=REFLECTIONS, batch=64
    )
    np.savez_compressed(TEST_CANDIDATES, **built)
    return built


def build_labels() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if LABEL_CACHE.is_file():
        data = np.load(LABEL_CACHE)
        return data["labels"], data["quality"], json.loads(str(data["report_json"].item()))
    component = np.load(v297.COMPONENT_CACHE)
    inverse = np.load(v304.SCATTERER_CACHE)
    matches = np.load(v305.MATCH_CACHE)
    labels = np.zeros((len(component["delay"]), PATH_COUNT), dtype=np.float32)
    los_component = inverse["los_component"].astype(np.int64)
    labels[:, 0] = component["weight"][np.arange(len(labels)), los_component]
    confidence = np.exp(
        -0.5
        * (
            (matches["length_error_bins"] / 0.50) ** 2
            + (matches["h_error_bins"] / 1.00) ** 2
        )
    ).astype(np.float32)
    contribution = matches["mass"] * confidence
    np.add.at(
        labels,
        (matches["point_id"], matches["candidate_id"].astype(np.int64) + 1),
        contribution,
    )
    represented = labels.sum(axis=1)
    quality = np.clip(represented, 0.0, 1.0).astype(np.float32)
    valid = represented > 1e-12
    labels[valid] /= represented[valid, None]
    labels[~valid, 0] = 1.0
    report = {
        "represented_mass": summarize(represented),
        "quality": summarize(quality),
        "active_candidate_count": summarize((labels > 1e-5).sum(axis=1)),
        "direct_mass": summarize(labels[:, 0]),
    }
    np.savez_compressed(
        LABEL_CACHE,
        labels=labels,
        quality=quality,
        report_json=np.asarray(json.dumps(report)),
    )
    return labels, quality, report


def make_features(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_candidates: dict[str, np.ndarray],
    test_candidates: dict[str, np.ndarray],
    patches: np.ndarray,
) -> tuple[v264.FeatureSet, v264.FeatureSet]:
    train_node, test_node = v257.build_node_features(train_pos[:, :2], test_pos[:, :2])
    old_reflections = v264.REFLECTIONS
    old_paths = v264.PATHS_PER_POINT
    try:
        v264.REFLECTIONS = REFLECTIONS
        v264.PATHS_PER_POINT = PATH_COUNT
        train = v264.build_features(train_pos, train_candidates, patches, train_node)
        test = v264.build_features(test_pos, test_candidates, patches, test_node)
    finally:
        v264.REFLECTIONS = old_reflections
        v264.PATHS_PER_POINT = old_paths
    return train, test


def knn_anchor(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    train_pos: np.ndarray,
    truth_d: np.ndarray,
) -> np.ndarray:
    tree = cKDTree(train_pos[bank_ids, :2])
    count = min(K_ANCHOR + 1, len(bank_ids))
    distance, local = tree.query(train_pos[query_ids, :2], k=count)
    if count == 1:
        distance = distance[:, None]
        local = local[:, None]
    selected_ids = np.empty((len(query_ids), min(K_ANCHOR, len(bank_ids))), dtype=np.int64)
    selected_distance = np.empty_like(selected_ids, dtype=np.float64)
    bank_lookup = bank_ids[local]
    for row, query_id in enumerate(query_ids):
        keep = bank_lookup[row] != query_id
        ids = bank_lookup[row, keep][: selected_ids.shape[1]]
        dist = distance[row, keep][: selected_ids.shape[1]]
        if len(ids) < selected_ids.shape[1]:
            ids = bank_lookup[row, : selected_ids.shape[1]]
            dist = distance[row, : selected_ids.shape[1]]
        selected_ids[row] = ids
        selected_distance[row] = dist
    weight = 1.0 / np.maximum(selected_distance, 0.05) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return v297.normalize_rows(
        np.sum(weight[:, :, None] * truth_d[selected_ids], axis=1, dtype=np.float64)
    )


class MatchedPatchEnergyTransformer(nn.Module):
    def __init__(self, global_dim: int, path_dim: int, patch_count: int):
        super().__init__()
        hidden = 128
        self.patch_embedding = nn.Embedding(patch_count + 1, 32)
        self.type_embedding = nn.Embedding(2, 8)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 256), nn.SiLU(), nn.LayerNorm(256), nn.Linear(256, hidden)
        )
        self.path_encoder = nn.Sequential(
            nn.Linear(path_dim + 40, 256), nn.SiLU(), nn.LayerNorm(256), nn.Linear(256, hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=384,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.energy_head = nn.Linear(hidden, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        nn.init.constant_(self.gate[-1].bias, -1.5)

    def forward(
        self,
        feature: v264.FeatureSet,
        ids: np.ndarray,
        anchor: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        global_feature = torch.from_numpy(feature.global_features[ids]).to(device)
        numeric = torch.from_numpy(feature.path_numeric[ids]).to(device)
        patch_id = torch.from_numpy(feature.patch_id[ids]).to(device)
        path_type = torch.from_numpy(feature.path_type[ids]).to(device)
        valid = torch.from_numpy(feature.path_valid[ids]).to(device)
        delay = torch.from_numpy(feature.delay[ids]).to(device)
        query = self.global_encoder(global_feature)
        path = self.path_encoder(
            torch.cat(
                [numeric, self.patch_embedding(patch_id), self.type_embedding(path_type)],
                dim=2,
            )
        )
        token = path + query[:, None, :]
        encoded = self.transformer(token, src_key_padding_mask=~valid)
        logits = self.energy_head(encoded).squeeze(2).masked_fill(~valid, -1e9)
        weights = torch.softmax(logits, dim=1)
        bins = torch.arange(v297.N_DELAY, device=device, dtype=delay.dtype)[None, None, :]
        delta = bins - delay[:, :, None]
        delta = torch.remainder(delta + v297.N_DELAY / 2.0, v297.N_DELAY) - v297.N_DELAY / 2.0
        atoms = torch.square(torch.sinc(delta) / torch.sinc(delta / v297.N_DELAY))
        atoms /= atoms.sum(dim=2, keepdim=True).clamp_min(1e-30)
        path_pdp = torch.sum(weights[:, :, None] * atoms, dim=1)
        pooled = torch.sum(weights[:, :, None] * encoded, dim=1)
        gate = 0.70 * torch.sigmoid(self.gate(torch.cat([query, pooled], dim=1)))
        prediction = (1.0 - gate) * anchor + gate * path_pdp
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {"weights": weights, "gate": gate.squeeze(1)}


MODEL_CLASS = MatchedPatchEnergyTransformer


def default_supervision_loss(
    diagnostics: dict[str, torch.Tensor],
    labels: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor:
    label_loss = -torch.sum(
        labels * torch.log(diagnostics["weights"].clamp_min(1e-10)), dim=1
    )
    return torch.sum(label_loss * quality) / quality.sum().clamp_min(1e-6)


SUPERVISION_LOSS = default_supervision_loss


@torch.no_grad()
def predict(
    model: MatchedPatchEnergyTransformer,
    feature: v264.FeatureSet,
    anchor: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty_like(anchor)
    gate = np.empty(len(anchor), dtype=np.float32)
    for start in range(0, len(anchor), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(anchor))
        ids = np.arange(start, end)
        current, diagnostics = model(
            feature, ids, torch.from_numpy(anchor[ids]).to(device), device
        )
        output[start:end] = current.cpu().numpy()
        gate[start:end] = diagnostics["gate"].cpu().numpy()
    return output, gate


def fit_model(
    train_feature: v264.FeatureSet,
    train_anchor: np.ndarray,
    train_truth: np.ndarray,
    train_labels: np.ndarray,
    train_quality: np.ndarray,
    seed: int,
    epochs: int,
    patch_count: int,
    device: torch.device,
    validation: tuple[v264.FeatureSet, np.ndarray, np.ndarray] | None = None,
) -> tuple[MatchedPatchEnergyTransformer, dict[str, object]]:
    set_seed(seed)
    model = MODEL_CLASS(
        train_feature.global_features.shape[1], train_feature.path_numeric.shape[2], patch_count
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_epoch = epochs
    best_score = -np.inf
    best_state = None
    history = []
    if validation is not None:
        val_feature, val_anchor, val_truth = validation
        best_epoch = 0
        best_score = float(np.mean(v297.row_cos(val_anchor, val_truth)))
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_truth))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            anchor = torch.from_numpy(train_anchor[ids]).to(device)
            truth = torch.from_numpy(train_truth[ids]).to(device)
            labels = torch.from_numpy(train_labels[ids]).to(device)
            quality = torch.from_numpy(train_quality[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, diagnostics = model(train_feature, ids, anchor, device)
            pdp_loss = (1.0 - F.cosine_similarity(prediction, truth, dim=1)).mean()
            label_loss = SUPERVISION_LOSS(diagnostics, labels, quality)
            loss = pdp_loss + 0.08 * label_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite v306 loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if validation is not None and (epoch % EVAL_EVERY == 0 or epoch == epochs):
            val_prediction, val_gate = predict(model, val_feature, val_anchor, device)
            score = float(np.mean(v297.row_cos(val_prediction, val_truth)))
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_pdp": score,
                    "validation_gate_mean": float(np.mean(val_gate)),
                }
            )
            if score > best_score:
                best_epoch = epoch
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if epoch % 20 == 0 or epoch == epochs:
            print(f"v306 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_validation_pdp": None if not np.isfinite(best_score) else float(best_score),
        "history": history,
    }


def choose_alpha(base: np.ndarray, model: np.ndarray, truth: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        prediction = v297.normalize_rows(base + float(alpha) * (model - base))
        rows.append({"alpha": float(alpha), "pdp": float(np.mean(v297.row_cos(prediction, truth)))})
    best = max(rows, key=lambda row: row["pdp"])
    return float(best["alpha"]), rows


def subset_feature(feature: v264.FeatureSet, ids: np.ndarray) -> v264.FeatureSet:
    return feature.take(ids)


def main() -> None:
    for path in (TRAIN_CANDIDATES, TARGET_CACHE, BASE, SCENE_CACHE, v305.MATCH_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    scene = np.load(SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    train_data = np.load(TRAIN_CANDIDATES)
    train_candidates = {key: train_data[key] for key in train_data.files}
    test_candidates = build_test_candidates(test_pos, patches)
    train_raw, test_raw = make_features(
        train_pos, test_pos, train_candidates, test_candidates, patches
    )
    labels, quality, label_report = build_labels()
    targets = np.load(TARGET_CACHE)
    truth_d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows = []
    base_parts = []
    model_parts = []
    truth_parts = []
    selected_epochs = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = v273.matched_subset(
            outer_fit, train_pos[:, :2], test_pos[:, :2], INNER_COUNT, seed + 10_000
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_fit_raw = subset_feature(train_raw, inner_fit)
        inner_val_raw = subset_feature(train_raw, inner_ids)
        inner_fit_feature, _full, inner_val_feature, _scale = v264.standardize_features(
            inner_fit_raw, train_raw, inner_val_raw
        )
        inner_train_anchor = knn_anchor(inner_fit, inner_fit, train_pos, truth_d)
        inner_val_anchor = knn_anchor(inner_ids, inner_fit, train_pos, truth_d)
        _model, inner_report = fit_model(
            inner_fit_feature,
            inner_train_anchor,
            truth_d[inner_fit],
            labels[inner_fit],
            quality[inner_fit],
            seed,
            EPOCHS,
            len(patches),
            device,
            (inner_val_feature, inner_val_anchor, truth_d[inner_ids]),
        )
        selected_epoch = int(inner_report["best_epoch"])
        selected_epochs.append(selected_epoch)

        outer_fit_raw = subset_feature(train_raw, outer_fit)
        outer_val_raw = subset_feature(train_raw, outer_ids)
        outer_fit_feature, _full, outer_val_feature, _scale = v264.standardize_features(
            outer_fit_raw, train_raw, outer_val_raw
        )
        outer_train_anchor = knn_anchor(outer_fit, outer_fit, train_pos, truth_d)
        outer_val_anchor = knn_anchor(outer_ids, outer_fit, train_pos, truth_d)
        if selected_epoch == 0:
            model_prediction = outer_val_anchor.copy()
            gate = np.zeros(len(model_prediction), dtype=np.float32)
            outer_training = {"best_epoch": 0}
        else:
            model, outer_training = fit_model(
                outer_fit_feature,
                outer_train_anchor,
                truth_d[outer_fit],
                labels[outer_fit],
                quality[outer_fit],
                seed + 50_000,
                selected_epoch,
                len(patches),
                device,
            )
            model_prediction, gate = predict(model, outer_val_feature, outer_val_anchor, device)
        base_score = float(np.mean(v297.row_cos(outer_val_anchor, truth_d[outer_ids])))
        model_score = float(np.mean(v297.row_cos(model_prediction, truth_d[outer_ids])))
        base_parts.append(outer_val_anchor)
        model_parts.append(model_prediction)
        truth_parts.append(truth_d[outer_ids])
        fold_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "selected_epoch": selected_epoch,
                "base_pdp": base_score,
                "model_pdp": model_score,
                "raw_delta": model_score - base_score,
                "gate": summarize(gate),
                "inner": inner_report,
                "outer_training": outer_training,
            }
        )
        print(
            f"v306 fold {fold + 1}/5 epoch={selected_epoch} base={base_score:.6f} "
            f"model={model_score:.6f} delta={model_score-base_score:+.6f}", flush=True
        )

    base_all = np.concatenate(base_parts)
    model_all = np.concatenate(model_parts)
    truth_all = np.concatenate(truth_parts)
    alpha, alpha_rows = choose_alpha(base_all, model_all, truth_all)
    blended = v297.normalize_rows(base_all + alpha * (model_all - base_all))
    base_score = float(np.mean(v297.row_cos(base_all, truth_all)))
    score = float(np.mean(v297.row_cos(blended, truth_all)))
    fold_deltas = []
    start = 0
    for row, base, model, truth in zip(fold_rows, base_parts, model_parts, truth_parts):
        end = start + len(base)
        current = blended[start:end]
        current_score = float(np.mean(v297.row_cos(current, truth)))
        row["blended_pdp"] = current_score
        row["blended_delta"] = current_score - float(row["base_pdp"])
        fold_deltas.append(float(row["blended_delta"]))
        start = end
    official_delta = score - base_score
    official_selection = bool(
        official_delta >= MIN_OFFICIAL_PDP_DELTA
        and sum(value > 0.0 for value in fold_deltas) >= 4
    )

    deployment = None
    if official_selection:
        full_raw = subset_feature(train_raw, valid_ids)
        full_feature, _unused, test_feature, _scale = v264.standardize_features(
            full_raw, train_raw, test_raw
        )
        train_anchor = knn_anchor(valid_ids, valid_ids, train_pos, truth_d)
        tree = cKDTree(train_pos[valid_ids, :2])
        distance, local = tree.query(test_pos[:, :2], k=K_ANCHOR)
        weight = 1.0 / np.maximum(distance, 0.05) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        test_anchor = v297.normalize_rows(
            np.sum(weight[:, :, None] * truth_d[valid_ids[local]], axis=1, dtype=np.float64)
        )
        final_epochs = max(1, int(np.median([value for value in selected_epochs if value > 0])))
        predictions = []
        checkpoints = []
        for seed in FINAL_SEEDS:
            model, _training = fit_model(
                full_feature,
                train_anchor,
                truth_d[valid_ids],
                labels[valid_ids],
                quality[valid_ids],
                seed,
                final_epochs,
                len(patches),
                device,
            )
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            checkpoints.append(PATHS.relative(checkpoint))
            prediction, _gate = predict(model, test_feature, test_anchor, device)
            predictions.append(prediction)
        model_test = v297.normalize_rows(np.mean(predictions, axis=0))
        target_d = v297.normalize_rows(test_anchor + alpha * (model_test - test_anchor))
        np.savez_compressed(TARGETS_OUT, d=target_d, anchor=test_anchor, model=model_test, alpha=np.float32(alpha))
        current = v225.current_delay_marginals(BASE)
        diagnostics = v225.apply_delay_targets(
            BASE,
            OUTPUT,
            {"d": target_d, "du": current["du"], "dh": current["dh"]},
            regime_of(test_pos[:, 2]),
            0.85,
        )
        deployment = {
            "final_epochs": final_epochs,
            "checkpoints": checkpoints,
            "prediction": PATHS.relative(OUTPUT),
            "targets": PATHS.relative(TARGETS_OUT),
            "local_score": v208.score_prediction(OUTPUT),
            "diagnostics": {key: summarize(value) for key, value in diagnostics.items()},
        }

    report = {
        "method": VERSION,
        "status": "official_selection_passed" if official_selection else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Absolute-delay and departure-angle matched wall-patch labels provide cleaner "
            "candidate existence/energy supervision than same-point NNLS weights."
        ),
        "label_report": label_report,
        "model_class": MODEL_CLASS.__name__,
        "fold_rows": fold_rows,
        "oof": {
            "base_pdp": base_score,
            "model_pdp": float(np.mean(v297.row_cos(model_all, truth_all))),
            "residual_alpha": alpha,
            "blended_pdp": score,
            "delta": official_delta,
            "fold_deltas": fold_deltas,
            "fold_wins": int(sum(value > 0.0 for value in fold_deltas)),
            "minimum_required_delta": MIN_OFFICIAL_PDP_DELTA,
            "official_selection": official_selection,
            "alpha_grid": alpha_rows,
        },
        "deployment": deployment,
        "compliance_boundary": {
            "uses": [
                "official train H path identity labels",
                "official pointcloud reflection candidates and shared patch IDs",
                "official train/test coordinates and scene features",
                "PyTorch patch-energy Transformer",
            ],
            "does_not_use": [
                "synthetic labels",
                "teacher outputs",
                "hidden labels",
                "RT coefficients or local truth for selection",
            ],
        },
        "outputs": {
            "test_candidates": PATHS.relative(TEST_CANDIDATES),
            "labels": PATHS.relative(LABEL_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["label_report"], indent=2), flush=True)
    print(json.dumps(report["oof"], indent=2), flush=True)


if __name__ == "__main__":
    main()
