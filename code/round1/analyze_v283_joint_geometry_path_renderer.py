#!/usr/bin/env python3
"""End-to-end joint component renderer from official wall-path geometry."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v265_scatterer_delay_angle_oracle as v265
import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 283
VERSION = "v283_joint_geometry_path_renderer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v283_joint_geometry_targets.npz"
OUTPUT = OUT_DIR / "v283_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
TEST_CANDIDATES = PATHS.cache_dir(264) / "v264_test_wall_reflection_candidates.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
V265_REPORT = PATHS.version_report_dir(265) / "v265_scatterer_delay_angle_oracle.json"

LOCAL_GATE = 0.645604
OUTER_SEEDS = (283, 1283, 2283, 3283, 4283)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 50
EVAL_EVERY = 5
BATCH_SIZE = 64
MIN_OFFICIAL_DELTA = 5e-4
FINAL_SEEDS = (28301, 28331, 28361)
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)
OFFSETS = tuple(
    (ds, dh) for ds in range(-3, 4) for dh in range(-3, 4)
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def knn_anchor(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    positions: np.ndarray,
    truth_sh: np.ndarray,
) -> np.ndarray:
    take = min(16, len(bank_ids) - int(np.array_equal(query_ids, bank_ids)))
    distance, local = cKDTree(positions[bank_ids, :2]).query(
        positions[query_ids, :2], k=take + int(np.array_equal(query_ids, bank_ids))
    )
    if distance.ndim == 1:
        distance = distance[:, None]
        local = local[:, None]
    absolute = bank_ids[local]
    selected = np.empty((len(query_ids), take), dtype=np.int64)
    selected_distance = np.empty((len(query_ids), take), dtype=np.float64)
    for row, query_id in enumerate(query_ids):
        keep = absolute[row] != query_id
        selected[row] = absolute[row, keep][:take]
        selected_distance[row] = distance[row, keep][:take]
    weight = 1.0 / np.maximum(selected_distance, 0.05) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return v279.normalize(
        np.sum(
            weight[:, :, None, None] * truth_sh[selected],
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)
    )


class JointPathRenderer(nn.Module):
    def __init__(self, global_dim: int, path_dim: int, patch_count: int):
        super().__init__()
        hidden = 160
        self.patch_embedding = nn.Embedding(patch_count + 1, 40)
        self.type_embedding = nn.Embedding(2, 12)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 320),
            nn.SiLU(),
            nn.LayerNorm(320),
            nn.Linear(320, hidden),
        )
        self.anchor_encoder = nn.Sequential(
            nn.Linear(v279.S_SIZE * v279.H_SIZE, 320),
            nn.SiLU(),
            nn.LayerNorm(320),
            nn.Linear(320, hidden),
        )
        self.path_encoder = nn.Sequential(
            nn.Linear(path_dim + 40 + 12 + 8, 320),
            nn.SiLU(),
            nn.LayerNorm(320),
            nn.Linear(320, hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=5,
            dim_feedforward=512,
            dropout=0.06,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.path_score = nn.Linear(hidden, 1)
        self.offset_score = nn.Linear(hidden, len(OFFSETS))
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3, 192), nn.SiLU(), nn.Linear(192, 1)
        )
        nn.init.constant_(self.gate[-1].bias, -1.7)
        self.register_buffer(
            "offset_prior",
            torch.tensor(
                [-0.70 * abs(ds) - 0.65 * abs(dh) for ds, dh in OFFSETS],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "offset_ds",
            torch.tensor([value[0] for value in OFFSETS], dtype=torch.long),
        )
        self.register_buffer(
            "offset_dh",
            torch.tensor([value[1] for value in OFFSETS], dtype=torch.long),
        )

    def forward(
        self,
        global_feature: torch.Tensor,
        path_numeric: torch.Tensor,
        patch_id: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        center_s: torch.Tensor,
        center_h: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, path_count = center_s.shape
        global_latent = self.global_encoder(global_feature)
        anchor_flat = anchor.reshape(batch, -1)
        anchor_latent = self.anchor_encoder(torch.sqrt(anchor_flat.clamp_min(0.0)))
        s_angle = 2.0 * math.pi * center_s.float() / v279.S_SIZE
        h_angle = 2.0 * math.pi * center_h.float() / v279.H_SIZE
        center_feature = torch.stack(
            [
                torch.sin(s_angle),
                torch.cos(s_angle),
                torch.sin(2.0 * s_angle),
                torch.cos(2.0 * s_angle),
                torch.sin(h_angle),
                torch.cos(h_angle),
                torch.sin(2.0 * h_angle),
                torch.cos(2.0 * h_angle),
            ],
            dim=2,
        )
        path = self.path_encoder(
            torch.cat(
                [
                    path_numeric,
                    self.patch_embedding(patch_id),
                    self.type_embedding(path_type),
                    center_feature,
                ],
                dim=2,
            )
        )
        query = global_latent + anchor_latent
        token = torch.cat([query[:, None, :], path], dim=1)
        mask = torch.cat(
            [
                torch.zeros((batch, 1), dtype=torch.bool, device=path.device),
                ~path_valid,
            ],
            dim=1,
        )
        encoded = self.transformer(token, src_key_padding_mask=mask)
        query_encoded = encoded[:, 0]
        path_encoded = encoded[:, 1:]
        path_logits = self.path_score(path_encoded).squeeze(2)
        path_logits = path_logits.masked_fill(~path_valid, -1e9)
        path_energy = torch.softmax(path_logits, dim=1)
        offset_delta = 2.0 * torch.tanh(self.offset_score(path_encoded))
        offset_probability = torch.softmax(
            self.offset_prior[None, None, :] + offset_delta, dim=2
        )
        destination = (
            torch.remainder(center_s[:, :, None] + self.offset_ds, v279.S_SIZE)
            * v279.H_SIZE
            + torch.remainder(center_h[:, :, None] + self.offset_dh, v279.H_SIZE)
        )
        sparse = torch.zeros(
            (batch, v279.S_SIZE * v279.H_SIZE),
            dtype=anchor.dtype,
            device=anchor.device,
        )
        sparse.scatter_add_(
            1,
            destination.reshape(batch, -1),
            (path_energy[:, :, None] * offset_probability).reshape(batch, -1),
        )
        pooled = torch.sum(path_energy[:, :, None] * path_encoded, dim=1)
        gate = 0.58 * torch.sigmoid(
            self.gate(torch.cat([query_encoded, anchor_latent, pooled], dim=1))
        )
        prediction = (1.0 - gate) * anchor_flat + gate * sparse
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {
            "gate": gate.squeeze(1),
            "path_energy": path_energy,
            "offset_probability": offset_probability,
        }


def loss_fn(
    prediction: torch.Tensor,
    truth_sh: torch.Tensor,
    truth_d: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> torch.Tensor:
    pred_d = v279.derive_d_torch(prediction)
    sh_loss = 1.0 - F.cosine_similarity(prediction, truth_sh, dim=1).mean()
    d_loss = 1.0 - F.cosine_similarity(pred_d, truth_d, dim=1).mean()
    zero_index = OFFSETS.index((0, 0))
    offset_penalty = (
        1.0 - diagnostics["offset_probability"][:, :, zero_index]
    ).mean()
    entropy = -torch.sum(
        diagnostics["path_energy"]
        * torch.log(diagnostics["path_energy"].clamp_min(1e-12)),
        dim=1,
    ).mean()
    return 0.52 * sh_loss + 0.48 * d_loss + 1e-4 * offset_penalty + 2e-5 * entropy


@torch.no_grad()
def predict(
    model: JointPathRenderer,
    feature: v264.FeatureSet,
    center_s: np.ndarray,
    center_h: np.ndarray,
    anchor: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty((len(anchor), v279.S_SIZE, v279.H_SIZE), dtype=np.float32)
    gates = np.empty(len(anchor), dtype=np.float32)
    for start in range(0, len(anchor), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(anchor))
        value, diagnostics = model(
            torch.from_numpy(feature.global_features[start:end]).to(device),
            torch.from_numpy(feature.path_numeric[start:end]).to(device),
            torch.from_numpy(feature.patch_id[start:end]).to(device),
            torch.from_numpy(feature.path_type[start:end]).to(device),
            torch.from_numpy(feature.path_valid[start:end]).to(device),
            torch.from_numpy(center_s[start:end]).to(device),
            torch.from_numpy(center_h[start:end]).to(device),
            torch.from_numpy(anchor[start:end]).to(device),
        )
        output[start:end] = value.reshape(-1, v279.S_SIZE, v279.H_SIZE).cpu().numpy()
        gates[start:end] = diagnostics["gate"].cpu().numpy()
    return v279.normalize(output), gates


def train_model(
    feature: v264.FeatureSet,
    center_s: np.ndarray,
    center_h: np.ndarray,
    anchor: np.ndarray,
    target_sh: np.ndarray,
    target_d: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    patch_count: int,
    validation: tuple[
        v264.FeatureSet,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    | None,
) -> tuple[JointPathRenderer, dict[str, object]]:
    set_seed(seed)
    model = JointPathRenderer(
        feature.global_features.shape[1], feature.path_numeric.shape[2], patch_count
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.8e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    rng = np.random.default_rng(seed)
    baseline = None
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    if validation is not None:
        _, _, _, val_anchor, val_sh, val_d = validation
        baseline = v279.metrics(val_anchor, val_sh, val_d)
        best_metric = baseline["weighted"]
        best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(anchor))
        losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            prediction, diagnostics = model(
                torch.from_numpy(feature.global_features[ids]).to(device),
                torch.from_numpy(feature.path_numeric[ids]).to(device),
                torch.from_numpy(feature.patch_id[ids]).to(device),
                torch.from_numpy(feature.path_type[ids]).to(device),
                torch.from_numpy(feature.path_valid[ids]).to(device),
                torch.from_numpy(center_s[ids]).to(device),
                torch.from_numpy(center_h[ids]).to(device),
                torch.from_numpy(anchor[ids]).to(device),
            )
            loss = loss_fn(
                prediction,
                torch.from_numpy(target_sh[ids].reshape(len(ids), -1)).to(device),
                torch.from_numpy(target_d[ids]).to(device),
                diagnostics,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs:
            item: dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
            }
            if validation is not None:
                val_feature, val_s, val_h, val_anchor, val_sh, val_d = validation
                val_pred, gates = predict(
                    model, val_feature, val_s, val_h, val_anchor, device
                )
                metric = v279.metrics(val_pred, val_sh, val_d)
                item["validation"] = metric
                item["baseline"] = baseline
                item["gate"] = summarize(gates)
                if metric["weighted"] > best_metric:
                    best_metric = metric["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"v283 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": float(best_metric) if np.isfinite(best_metric) else None,
        "baseline": baseline,
        "history": history,
    }


def centers(
    positions: np.ndarray,
    candidates: dict[str, np.ndarray],
    feature: v264.FeatureSet,
    spec: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    azimuth, _valid, _patch_id = v265.departure_azimuth(positions, candidates)
    h = v265.h_center_for(
        azimuth,
        float(spec["orientation"]),
        str(spec["h_mode"]),
        float(spec["h_sign"]),
        float(spec["h_shift"]),
    )
    s = np.mod(float(spec["s_sign"]) * feature.delay, v279.S_SIZE)
    return (
        np.rint(s).astype(np.int64) % v279.S_SIZE,
        np.rint(h).astype(np.int64) % v279.H_SIZE,
    )


def main() -> None:
    for path in (
        TARGET_CACHE,
        TRAIN_CANDIDATES,
        TEST_CANDIDATES,
        SCENE_CACHE,
        V265_REPORT,
        BASE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_data = np.load(TARGET_CACHE)
    truth_sh = v279.normalize(target_data["sh"].astype(np.float32))
    truth_d = v279.normalize(target_data["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    patches = np.load(SCENE_CACHE)["wall_patches"].astype(np.float32)
    train_candidates = dict(np.load(TRAIN_CANDIDATES))
    test_candidates = dict(np.load(TEST_CANDIDATES))
    train_node, test_node = v257.build_node_features(
        train_pos[:, :2], test_pos[:, :2]
    )
    train_raw = v264.build_features(
        train_pos, train_candidates, patches, train_node
    )
    test_raw = v264.build_features(test_pos, test_candidates, patches, test_node)
    with V265_REPORT.open("r", encoding="utf-8") as handle:
        best_spec = json.load(handle)["selection"]["best"]
    train_s, train_h = centers(
        train_pos, train_candidates, train_raw, best_spec
    )
    test_s, test_h = centers(test_pos, test_candidates, test_raw, best_spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v283 device={device} valid={len(valid_ids)}", flush=True)

    outer_rows: list[dict[str, object]] = []
    oof_ids_parts: list[np.ndarray] = []
    oof_base_parts: list[np.ndarray] = []
    oof_model_parts: list[np.ndarray] = []
    selected_epochs: list[int] = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], INNER_COUNT, seed + 50_000
            ),
            outer_fit,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)

        inner_fit_raw = train_raw.take(inner_fit)
        inner_feature, inner_full, _inner_test, _ = v264.standardize_features(
            inner_fit_raw, train_raw, test_raw
        )
        inner_val_feature = inner_full.take(inner_ids)
        inner_fit_anchor = knn_anchor(
            inner_fit, inner_fit, train_pos, truth_sh
        )
        inner_val_anchor = knn_anchor(
            inner_ids, inner_fit, train_pos, truth_sh
        )
        _, inner_training = train_model(
            inner_feature,
            train_s[inner_fit],
            train_h[inner_fit],
            inner_fit_anchor,
            truth_sh[inner_fit],
            truth_d[inner_fit],
            seed,
            EPOCHS,
            device,
            len(patches),
            (
                inner_val_feature,
                train_s[inner_ids],
                train_h[inner_ids],
                inner_val_anchor,
                truth_sh[inner_ids],
                truth_d[inner_ids],
            ),
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)

        outer_fit_raw = train_raw.take(outer_fit)
        outer_feature, outer_full, _outer_test, _ = v264.standardize_features(
            outer_fit_raw, train_raw, test_raw
        )
        outer_val_feature = outer_full.take(outer_ids)
        outer_fit_anchor = knn_anchor(
            outer_fit, outer_fit, train_pos, truth_sh
        )
        outer_val_anchor = knn_anchor(
            outer_ids, outer_fit, train_pos, truth_sh
        )
        if selected_epoch == 0:
            model_sh = outer_val_anchor.copy()
            gates = np.zeros(len(outer_ids), dtype=np.float32)
            outer_training = {"seed": seed + 100_000, "best_epoch": 0, "history": []}
        else:
            model, outer_training = train_model(
                outer_feature,
                train_s[outer_fit],
                train_h[outer_fit],
                outer_fit_anchor,
                truth_sh[outer_fit],
                truth_d[outer_fit],
                seed + 100_000,
                selected_epoch,
                device,
                len(patches),
                None,
            )
            model_sh, gates = predict(
                model,
                outer_val_feature,
                train_s[outer_ids],
                train_h[outer_ids],
                outer_val_anchor,
                device,
            )
        baseline = v279.metrics(
            outer_val_anchor, truth_sh[outer_ids], truth_d[outer_ids]
        )
        model_metric = v279.metrics(
            model_sh, truth_sh[outer_ids], truth_d[outer_ids]
        )
        delta = {
            key: model_metric[key] - baseline[key]
            for key in ("sh", "d", "weighted")
        }
        outer_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "selected_epoch": selected_epoch,
                "inner_training": inner_training,
                "outer_training": outer_training,
                "baseline": baseline,
                "model": model_metric,
                "raw_delta": delta,
                "gate": summarize(gates),
            }
        )
        oof_ids_parts.append(outer_ids)
        oof_base_parts.append(outer_val_anchor)
        oof_model_parts.append(model_sh)
        print(
            f"v283 outer {fold + 1}/{len(OUTER_SEEDS)} epoch={selected_epoch} "
            f"base={baseline['weighted']:.6f} model={model_metric['weighted']:.6f} "
            f"delta={delta['weighted']:+.6f}",
            flush=True,
        )

    oof_ids = np.concatenate(oof_ids_parts)
    oof_base = np.concatenate(oof_base_parts)
    oof_model = np.concatenate(oof_model_parts)
    alpha, alpha_grid = v279.choose_joint_alpha(
        oof_base, oof_model, truth_sh[oof_ids], truth_d[oof_ids]
    )
    fold_delta: list[dict[str, float]] = []
    for row, ids, base_sh, model_sh in zip(
        outer_rows, oof_ids_parts, oof_base_parts, oof_model_parts
    ):
        blended = v279.normalize(base_sh + alpha * (model_sh - base_sh))
        metric = v279.metrics(blended, truth_sh[ids], truth_d[ids])
        delta = {
            key: metric[key] - row["baseline"][key]
            for key in ("sh", "d", "weighted")
        }
        row["blended"] = metric
        row["blended_delta"] = delta
        fold_delta.append(delta)
    mean_delta = {
        key: float(np.mean([value[key] for value in fold_delta]))
        for key in ("sh", "d", "weighted")
    }
    weighted_wins = int(np.sum([x["weighted"] > 0.0 for x in fold_delta]))
    official_selected = bool(
        alpha > 0.0
        and mean_delta["weighted"] >= MIN_OFFICIAL_DELTA
        and weighted_wins >= 3
        and mean_delta["sh"] > 0.0
        and mean_delta["d"] > 0.0
    )

    final_training: list[dict[str, object]] = []
    local_score = None
    projection_diagnostics = None
    if official_selected:
        positive_epochs = [epoch for epoch in selected_epochs if epoch > 0]
        final_epoch = max(int(np.median(positive_epochs)), 1)
        all_raw = train_raw.take(valid_ids)
        all_feature, _full, final_test_feature, _ = v264.standardize_features(
            all_raw, train_raw, test_raw
        )
        final_anchor = knn_anchor(valid_ids, valid_ids, train_pos, truth_sh)
        test_ids_dummy = np.arange(len(test_pos), dtype=np.int64)
        distance, local = cKDTree(train_pos[valid_ids, :2]).query(
            test_pos[:, :2], k=16
        )
        weight = 1.0 / np.maximum(distance, 0.05) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        test_anchor = v279.normalize(
            np.sum(
                weight[:, :, None, None] * truth_sh[valid_ids[local]],
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)
        )
        model_parts: list[np.ndarray] = []
        gate_parts: list[np.ndarray] = []
        for seed in FINAL_SEEDS:
            model, training = train_model(
                all_feature,
                train_s[valid_ids],
                train_h[valid_ids],
                final_anchor,
                truth_sh[valid_ids],
                truth_d[valid_ids],
                seed,
                final_epoch,
                device,
                len(patches),
                None,
            )
            test_model, test_gate = predict(
                model,
                final_test_feature,
                test_s,
                test_h,
                test_anchor,
                device,
            )
            model_parts.append(test_model)
            gate_parts.append(test_gate)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "model_state": model.state_dict(),
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
        test_model = v279.normalize(np.mean(model_parts, axis=0))
        current_sh, _ = v274.extract_current_marginals(BASE)
        residual = test_model - test_anchor
        target_sh = v279.normalize(current_sh + alpha * residual)
        target_d = v279.derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            anchor_sh=test_anchor,
            model_sh=test_model,
            current_sh=current_sh,
            target_sh=target_sh,
            target_d=target_d,
            residual=residual,
            alpha=np.asarray(alpha, dtype=np.float32),
            gate=np.mean(gate_parts, axis=0).astype(np.float32),
        )
        projection_diagnostics = v247.project(target_sh, target_d, OUTPUT)
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": (
            "local_gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Material/path-energy amplitudes do not transfer to the local gate, but full "
            "official wall-path geometry supplies nearly complete joint delay-angle component "
            "candidates whose occupancy and local offsets can be learned end-to-end."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H joint S-H/PDP labels",
                "official train/test coordinates",
                "official point-cloud full wall reflection geometry",
                "strict v239 official-only carrier after official selection",
            ],
            "does_not_use": [
                "NNLS/path-energy labels from v239/v264",
                "synthetic/Sionna channel labels",
                "v212/v217 teacher",
                "hidden labels or RT H/path coefficients",
                "local replay truth for training or selection",
            ],
            "local_truth_usage": (
                "one final gate score after nested official selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "path_count": v264.PATHS_PER_POINT,
            "offsets": [list(value) for value in OFFSETS],
            "epochs": EPOCHS,
            "alpha_grid": ALPHA_GRID.tolist(),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
            "final_seeds": list(FINAL_SEEDS),
        },
        "path_center_spec": best_spec,
        "outer_folds": outer_rows,
        "official_selection": {
            "residual_alpha": alpha,
            "alpha_grid": alpha_grid,
            "mean_delta": mean_delta,
            "weighted_wins": weighted_wins,
            "selected_epochs": selected_epochs,
            "selected": official_selected,
        },
        "final_training": final_training,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in projection_diagnostics.items()}
            if projection_diagnostics is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(
                    float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
                ),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
