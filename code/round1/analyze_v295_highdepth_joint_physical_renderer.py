#!/usr/bin/env python3
"""v295: joint angle-delay renderer from depth-16 interaction geometry."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
import analyze_v293_interaction_physical_path_law as v293
import analyze_v294_heteroscedastic_planar_gp as v294
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 295
VERSION = "v295_highdepth_joint_physical_renderer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v295_joint_physical_targets.npz"
OUTPUT = OUT_DIR / "v295_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
PATH_CACHE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
INTERACTION_CACHE = PATHS.cache_dir(293) / "v293_interaction_features.npz"
BARRIER_CACHE = PATHS.cache_dir(273) / "v273_planar_barrier_features.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"

LOCAL_GATE = 0.645604
OUTER_SEEDS = (295, 1295, 2295, 3295, 4295)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 60
FINAL_SEEDS = (29501, 29531, 29561)
MIN_OFFICIAL_DELTA = 5e-4
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)
BATCH_SIZE = 40
MAX_PATHS = 128
S_SHIFTS = (-0.85, 0.0, 0.85)
H_OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0)
GP_KERNEL = v294.Kernel(6.5, 0.08, 0.0, 0.0, 0.0, True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape(
        (int(np.sum(valid)),) + (1,) * (out.ndim - 1)
    )
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return v279.row_cos(a, b)


def metrics(prediction: np.ndarray, truth_sh: np.ndarray, truth_d: np.ndarray) -> dict[str, float]:
    return v279.metrics(prediction, truth_sh, truth_d)


@dataclass
class JointData:
    physical: v293.Prepared
    phi_t: np.ndarray

    def take(self, ids: np.ndarray) -> "JointData":
        return JointData(self.physical.take(ids), self.phi_t[ids])


def standardize(train: JointData, others: list[JointData]) -> tuple[JointData, list[JointData]]:
    train_physical, other_physical = v293.standardize(
        train.physical, [item.physical for item in others]
    )
    return (
        JointData(train_physical, train.phi_t),
        [JointData(value, source.phi_t) for value, source in zip(other_physical, others)],
    )


def gp_anchor(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    query_pos: np.ndarray,
    bank_pos: np.ndarray,
    query_signature: np.ndarray,
    bank_signature: np.ndarray,
    query_path: np.ndarray,
    bank_path: np.ndarray,
    bank_sh: np.ndarray,
    exclude_self: bool,
) -> np.ndarray:
    data = v294.neighborhood(
        query_ids,
        bank_ids,
        query_pos,
        bank_pos,
        query_signature,
        bank_signature,
        query_path,
        bank_path,
        exclude_self,
    )
    weight = v294.kriging_weights(
        data, GP_KERNEL, np.zeros(len(bank_pos), dtype=np.float32)
    )
    return v294.weighted_bank(bank_sh, data.indices, weight)


class JointPhysicalRenderer(nn.Module):
    def __init__(self, path_dim: int, summary_dim: int):
        super().__init__()
        self.raw_gamma = nn.Parameter(torch.tensor(0.55))
        self.raw_wall_loss = nn.Parameter(torch.tensor(-0.25))
        self.raw_ground_loss = nn.Parameter(torch.tensor(-0.65))
        self.raw_refraction_loss = nn.Parameter(torch.tensor(-0.15))
        self.path_residual = nn.Sequential(
            nn.Linear(path_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.s_head = nn.Sequential(
            nn.Linear(path_dim, 48), nn.SiLU(), nn.Linear(48, len(S_SHIFTS))
        )
        self.h_head = nn.Sequential(
            nn.Linear(path_dim, 48), nn.SiLU(), nn.Linear(48, len(H_OFFSETS))
        )
        self.gate_head = nn.Sequential(
            nn.Linear(summary_dim, 48), nn.SiLU(), nn.Linear(48, 1)
        )
        self.s_bias = nn.Parameter(torch.tensor([-0.25, 0.0, -0.25]))
        self.h_bias = nn.Parameter(torch.tensor([-1.5, -0.7, 0.0, -0.7, -1.5]))
        nn.init.zeros_(self.path_residual[-1].weight)
        nn.init.zeros_(self.path_residual[-1].bias)
        nn.init.zeros_(self.s_head[-1].weight)
        nn.init.zeros_(self.s_head[-1].bias)
        nn.init.zeros_(self.h_head[-1].weight)
        nn.init.zeros_(self.h_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -1.5)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        phi_t: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = 1.0 + F.softplus(self.raw_gamma)
        wall_loss = F.softplus(self.raw_wall_loss)
        ground_loss = F.softplus(self.raw_ground_loss)
        refraction_loss = F.softplus(self.raw_refraction_loss)
        wall_count = torch.clamp(
            batch["reflection_count"] - batch["ground_reflection"], min=0.0
        )
        score = (
            -gamma * batch["relative_log_length"]
            - wall_loss * wall_count * batch["reflection_mean"]
            - ground_loss * batch["ground_reflection"] * batch["reflection_mean"]
            - refraction_loss * batch["refraction_count"]
            + 0.85 * torch.tanh(self.path_residual(batch["path_features"]).squeeze(-1))
        )
        score = score.masked_fill(~batch["valid"], -1e4)
        path_weight = torch.softmax(score, dim=1) * batch["valid"].float()
        path_weight = path_weight / path_weight.sum(dim=1, keepdim=True).clamp_min(1e-20)
        s_probability = torch.softmax(
            self.s_bias[None, None] + 0.55 * torch.tanh(self.s_head(batch["path_features"])),
            dim=2,
        )
        h_probability = torch.softmax(
            self.h_bias[None, None] + 0.55 * torch.tanh(self.h_head(batch["path_features"])),
            dim=2,
        )

        s_values = torch.arange(v279.S_SIZE, device=anchor.device, dtype=torch.float32)
        h_values = torch.arange(v279.H_SIZE, device=anchor.device, dtype=torch.float32)
        s_shift = torch.tensor(S_SHIFTS, device=anchor.device, dtype=torch.float32)
        h_offset = torch.tensor(H_OFFSETS, device=anchor.device, dtype=torch.float32)
        s_center = torch.remainder(
            -batch["delay"][..., None] + s_shift[None, None], v279.S_SIZE
        )
        s_delta = torch.abs(s_values[None, None, None] - s_center[..., None])
        s_delta = torch.minimum(s_delta, v279.S_SIZE - s_delta)
        s_atoms = torch.exp(-0.5 * (s_delta / 0.75) ** 2)
        s_atoms = s_atoms / s_atoms.sum(dim=3, keepdim=True).clamp_min(1e-20)
        s_kernel = torch.sum(s_probability[..., None] * s_atoms, dim=2)

        base_h = torch.remainder(
            8.0 * torch.sin(phi_t + math.radians(30.0)), v279.H_SIZE
        )
        h_center = torch.remainder(
            base_h[..., None] + h_offset[None, None], v279.H_SIZE
        )
        h_delta = torch.abs(h_values[None, None, None] - h_center[..., None])
        h_delta = torch.minimum(h_delta, v279.H_SIZE - h_delta)
        h_atoms = torch.exp(-0.5 * (h_delta / 0.75) ** 2)
        h_atoms = h_atoms / h_atoms.sum(dim=3, keepdim=True).clamp_min(1e-20)
        h_kernel = torch.sum(h_probability[..., None] * h_atoms, dim=2)
        path_sh = torch.einsum("bp,bps,bph->bsh", path_weight, s_kernel, h_kernel)
        path_sh = path_sh / path_sh.sum(dim=(1, 2), keepdim=True).clamp_min(1e-20)
        has_path = batch["valid"].any(dim=1).float()
        gate = 0.40 * torch.sigmoid(self.gate_head(batch["summary"]).squeeze(1)) * has_path
        model = (1.0 - gate[:, None, None]) * anchor + gate[:, None, None] * path_sh
        model = model / model.sum(dim=(1, 2), keepdim=True).clamp_min(1e-20)
        return model, path_sh, gate


def tensor_batch(data: JointData, ids: np.ndarray, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch = v293.tensor_batch(data.physical, ids, device)
    phi = torch.from_numpy(data.phi_t[ids]).to(device=device, dtype=torch.float32)
    return batch, phi


def predict(
    model: JointPhysicalRenderer,
    data: JointData,
    anchor: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model_parts = []
    path_parts = []
    gate_parts = []
    with torch.no_grad():
        for start in range(0, len(anchor), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(anchor))
            ids = np.arange(start, end)
            batch, phi = tensor_batch(data, ids, device)
            anchor_t = torch.from_numpy(anchor[start:end]).to(device)
            model_sh, path_sh, gate = model(batch, phi, anchor_t)
            model_parts.append(model_sh.cpu().numpy().astype(np.float32))
            path_parts.append(path_sh.cpu().numpy().astype(np.float32))
            gate_parts.append(gate.cpu().numpy().astype(np.float32))
    return np.concatenate(model_parts), np.concatenate(path_parts), np.concatenate(gate_parts)


def loss_fn(
    model_sh: torch.Tensor,
    path_sh: torch.Tensor,
    truth_sh: torch.Tensor,
    truth_d: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    flat_model = model_sh.flatten(1)
    flat_path = path_sh.flatten(1)
    flat_truth = truth_sh.flatten(1)
    model_d = v279.derive_d_torch(flat_model)
    path_d = v279.derive_d_torch(flat_path)
    main = (
        0.52 * (1.0 - F.cosine_similarity(flat_model, flat_truth, dim=1).mean())
        + 0.48 * (1.0 - F.cosine_similarity(model_d, truth_d, dim=1).mean())
    )
    auxiliary = (
        0.52 * (1.0 - F.cosine_similarity(flat_path, flat_truth, dim=1).mean())
        + 0.48 * (1.0 - F.cosine_similarity(path_d, truth_d, dim=1).mean())
    )
    return main + 0.04 * auxiliary + 0.002 * torch.mean(gate**2)


def train_model(
    train: JointData,
    train_anchor: np.ndarray,
    train_truth_sh: np.ndarray,
    train_truth_d: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[JointData, np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[JointPhysicalRenderer, dict[str, object]]:
    set_seed(seed)
    model = JointPhysicalRenderer(
        train.physical.path_features.shape[2], train.physical.summary.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2e-4)
    order = np.arange(len(train_anchor))
    best_score = -np.inf
    best_epoch = epochs
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(order)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            batch, phi = tensor_batch(train, ids, device)
            anchor_t = torch.from_numpy(train_anchor[ids]).to(device)
            truth_sh_t = torch.from_numpy(train_truth_sh[ids]).to(device)
            truth_d_t = torch.from_numpy(train_truth_d[ids]).to(device)
            model_sh, path_sh, gate = model(batch, phi, anchor_t)
            if not (
                torch.isfinite(model_sh).all()
                and torch.isfinite(path_sh).all()
                and torch.isfinite(gate).all()
            ):
                raise FloatingPointError(
                    f"non-finite forward output seed={seed} epoch={epoch} batch={start}"
                )
            loss = loss_fn(model_sh, path_sh, truth_sh_t, truth_d_t, gate)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss seed={seed} epoch={epoch} batch={start}"
                )
            optimizer.zero_grad(set_to_none=True)
            if seed == OUTER_SEEDS[0] and epoch == 1 and start == 0:
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(
                        f"non-finite gradient {name} seed={seed} epoch={epoch} batch={start}"
                    )
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if validation is not None:
                val_data, val_anchor, val_sh, val_d = validation
                prediction, path_prediction, gate = predict(model, val_data, val_anchor, device)
                value = metrics(prediction, val_sh, val_d)
                item.update(
                    {
                        "validation": value,
                        "path": metrics(path_prediction, val_sh, val_d),
                        "gate_mean": float(np.mean(gate)),
                    }
                )
                if value["weighted"] > best_score:
                    best_score = value["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
            print(f"v295 seed={seed} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": None if not np.isfinite(best_score) else float(best_score),
        "history": history,
    }


def choose_alpha(base: np.ndarray, model: np.ndarray, truth_sh: np.ndarray, truth_d: np.ndarray) -> tuple[float, list[dict[str, object]]]:
    rows = []
    for alpha in ALPHA_GRID:
        prediction = normalize(base + float(alpha) * (model - base))
        rows.append({"alpha": float(alpha), **metrics(prediction, truth_sh, truth_d)})
    return float(max(rows, key=lambda item: float(item["weighted"]))["alpha"]), rows


def main() -> None:
    for path in (TARGET_CACHE, PATH_CACHE, BARRIER_CACHE, BASE):
        if not path.is_file():
            raise FileNotFoundError(path)
    path_cache = np.load(PATH_CACHE)
    if not INTERACTION_CACHE.is_file():
        v293.build_feature_cache(path_cache)
    interaction = np.load(INTERACTION_CACHE)
    train_physical = v293.prepared(dict(interaction), "train")
    test_physical = v293.prepared(dict(interaction), "test")
    train_data_all = JointData(
        train_physical,
        np.nan_to_num(path_cache["train_phi_t"].astype(np.float32), nan=0.0),
    )
    test_data_all = JointData(
        test_physical,
        np.nan_to_num(path_cache["test_phi_t"].astype(np.float32), nan=0.0),
    )
    targets = np.load(TARGET_CACHE)
    truth_sh = normalize(targets["sh"].astype(np.float32))
    truth_d = normalize(targets["d"].astype(np.float32))
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    barrier = np.load(BARRIER_CACHE)
    train_signature = barrier["train_signature"].astype(bool)
    test_signature = barrier["test_signature"].astype(bool)
    train_path = interaction["train_summary"].astype(np.float32)[:, 3:]
    test_path = interaction["test_summary"].astype(np.float32)[:, 3:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outer_rows = []
    oof_ids_parts = []
    oof_anchor_parts = []
    oof_model_parts = []
    selected_epochs = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(train_pos[:, :2], test_pos[:, :2], INNER_COUNT, seed + 50_000),
            outer_fit,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_train_anchor = gp_anchor(
            inner_fit, inner_fit, train_pos, train_pos, train_signature, train_signature,
            train_path, train_path, truth_sh, True,
        )
        inner_val_anchor = gp_anchor(
            inner_ids, inner_fit, train_pos, train_pos, train_signature, train_signature,
            train_path, train_path, truth_sh, False,
        )
        inner_train_raw = train_data_all.take(inner_fit)
        inner_val_raw = train_data_all.take(inner_ids)
        inner_train, [inner_val] = standardize(inner_train_raw, [inner_val_raw])
        _, inner_training = train_model(
            inner_train, inner_train_anchor, truth_sh[inner_fit], truth_d[inner_fit],
            seed, EPOCHS, device, (inner_val, inner_val_anchor, truth_sh[inner_ids], truth_d[inner_ids]),
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)

        outer_train_anchor = gp_anchor(
            outer_fit, outer_fit, train_pos, train_pos, train_signature, train_signature,
            train_path, train_path, truth_sh, True,
        )
        outer_val_anchor = gp_anchor(
            outer_ids, outer_fit, train_pos, train_pos, train_signature, train_signature,
            train_path, train_path, truth_sh, False,
        )
        outer_train_raw = train_data_all.take(outer_fit)
        outer_val_raw = train_data_all.take(outer_ids)
        outer_train, [outer_val] = standardize(outer_train_raw, [outer_val_raw])
        model, outer_training = train_model(
            outer_train, outer_train_anchor, truth_sh[outer_fit], truth_d[outer_fit],
            seed + 100_000, selected_epoch, device, None,
        )
        model_sh, path_sh, gate = predict(model, outer_val, outer_val_anchor, device)
        baseline = metrics(outer_val_anchor, truth_sh[outer_ids], truth_d[outer_ids])
        model_metric = metrics(model_sh, truth_sh[outer_ids], truth_d[outer_ids])
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
                "raw_delta": model_metric["weighted"] - baseline["weighted"],
                "path": metrics(path_sh, truth_sh[outer_ids], truth_d[outer_ids]),
                "gate": summarize(gate),
            }
        )
        oof_ids_parts.append(outer_ids)
        oof_anchor_parts.append(outer_val_anchor)
        oof_model_parts.append(model_sh)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"v295 outer {fold + 1}/5 epoch={selected_epoch} "
            f"delta={model_metric['weighted'] - baseline['weighted']:+.6f}",
            flush=True,
        )

    oof_ids = np.concatenate(oof_ids_parts)
    oof_anchor = np.concatenate(oof_anchor_parts)
    oof_model = np.concatenate(oof_model_parts)
    alpha, alpha_curve = choose_alpha(oof_anchor, oof_model, truth_sh[oof_ids], truth_d[oof_ids])
    fold_delta = []
    for row, ids, anchor, model in zip(outer_rows, oof_ids_parts, oof_anchor_parts, oof_model_parts):
        prediction = normalize(anchor + alpha * (model - anchor))
        value = metrics(prediction, truth_sh[ids], truth_d[ids])
        delta = value["weighted"] - float(row["baseline"]["weighted"])
        row["blended"] = value
        row["blended_delta"] = delta
        fold_delta.append(delta)
    mean_delta = float(np.mean(fold_delta))
    wins = int(np.sum(np.asarray(fold_delta) > 0.0))
    selected = bool(mean_delta >= MIN_OFFICIAL_DELTA and wins >= 3 and alpha > 0.0)

    final_training = []
    projection = None
    local = None
    if selected:
        full_anchor = gp_anchor(
            valid_ids, valid_ids, train_pos, train_pos, train_signature, train_signature,
            train_path, train_path, truth_sh, True,
        )
        test_anchor = gp_anchor(
            np.arange(len(test_pos)), valid_ids, test_pos, train_pos, test_signature, train_signature,
            test_path, train_path, truth_sh, False,
        )
        full_raw = train_data_all.take(valid_ids)
        full_train, [test_data] = standardize(full_raw, [test_data_all])
        final_epoch = int(np.median(selected_epochs))
        model_parts = []
        gate_parts = []
        for seed in FINAL_SEEDS:
            model, training = train_model(
                full_train, full_anchor, truth_sh[valid_ids], truth_d[valid_ids],
                seed, final_epoch, device, None,
            )
            model_sh, _, gate = predict(model, test_data, test_anchor, device)
            model_parts.append(model_sh)
            gate_parts.append(gate)
            final_training.append(training)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        test_model = normalize(np.mean(model_parts, axis=0))
        current_sh, _ = v274.extract_current_marginals(BASE)
        target_sh = normalize(current_sh + alpha * (test_model - test_anchor))
        target_d = v279.derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            anchor_sh=test_anchor,
            model_sh=test_model,
            current_sh=current_sh,
            target_sh=target_sh,
            target_d=target_d,
            alpha=np.asarray(alpha, dtype=np.float32),
            gate=np.mean(gate_parts, axis=0).astype(np.float32),
        )
        projection_raw = v247.project(target_sh, target_d, OUTPUT)
        projection = {key: summarize(value) for key, value in projection_raw.items()}
        local = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A depth-16 path must create one coupled delay/BS-angle component; rendering "
            "joint S-H with the verified S=(-D), H=8*sin(phi+30deg) mapping exposes "
            "clean Sionna structure missing from separate marginal interpolation."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H joint S-H/PDP",
                "official train/test planar coordinates and point cloud",
                "v291 geometry-only path delays, departure angles, vertices and interactions",
                "official-only Matérn anchor frozen from v294",
                "strict v239 carrier after official selection",
            ],
            "does_not_use": [
                "Paths.cfr or path coefficients",
                "synthetic/Sionna channel labels",
                "v212/v217 teacher output",
                "hidden labels or local truth for selection",
            ],
            "local_truth_usage": "read once after official selection" if selected else "not read",
        },
        "parameters": {
            "epochs": EPOCHS,
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "gp_anchor": GP_KERNEL.name,
            "s_shifts": list(S_SHIFTS),
            "h_offsets": list(H_OFFSETS),
            "alpha_grid": ALPHA_GRID.tolist(),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
        },
        "outer_folds": outer_rows,
        "official_selection": {
            "residual_alpha": alpha,
            "alpha_curve": alpha_curve,
            "fold_deltas": fold_delta,
            "mean_weighted_delta": mean_delta,
            "wins": wins,
            "selected_epochs": selected_epochs,
            "selected": selected,
        },
        "final_training": final_training,
        "projection": projection,
        "local_gate": local,
        "outputs": {
            "targets": PATHS.relative(TARGETS_OUT) if selected else None,
            "prediction": PATHS.relative(OUTPUT) if selected else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
