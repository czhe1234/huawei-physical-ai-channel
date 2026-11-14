#!/usr/bin/env python3
"""DETR-style continuous path-set transport with Hungarian supervision."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import analyze_v254_route_topology_spectral_field as v254
import analyze_v273_planar_barrier_kernel as v273
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 303
VERSION = "v303_hungarian_path_set_transport"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

OUTER_SEEDS = (303, 1303, 2303, 3303, 4303)
OUTER_COUNT = 320
INNER_COUNT = 240
SLOT_COUNT = v297.COMPONENT_COUNT
EPOCHS = 60
EVAL_EVERY = 5
BATCH_SIZE = 32
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)
MIN_OFFICIAL_PDP_DELTA = 0.01


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_arrays(
    data: v274.ModelSet,
    node_standardized: np.ndarray,
    components: v297.ComponentData,
    truth_d: np.ndarray,
    fingerprint: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    arrays = v297.model_arrays(data, node_standardized, components, truth_d)
    neighbor = data.neighbor_ids
    arrays.update(
        {
            "fingerprint_horizontal": fingerprint["horizontal"][neighbor].astype(np.float32),
            "fingerprint_vertical": fingerprint["vertical"][neighbor].astype(np.float32),
            "fingerprint_polarization": fingerprint["polarization"][neighbor].astype(np.float32),
            "fingerprint_ue": fingerprint["ue_port"][neighbor].astype(np.float32),
            "fingerprint_gain_power": fingerprint["gain_power"][neighbor].astype(np.float32),
        }
    )
    return arrays


def tensor_batch(
    arrays: dict[str, np.ndarray], rows: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.asarray(value[rows])).to(device)
        for key, value in arrays.items()
        if key != "query_ids"
    }


class HungarianPathSetTransport(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        hidden = 96
        self.query_encoder = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, hidden)
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, hidden)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 96), nn.SiLU(), nn.Linear(96, 48)
        )
        self.component_encoder = nn.Sequential(
            nn.Linear(44, 128), nn.SiLU(), nn.Linear(128, 48)
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(hidden * 3 + 96, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, hidden),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=256,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=256,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.path_slots = nn.Parameter(torch.randn(SLOT_COUNT, hidden) * 0.04)
        self.delay_head = nn.Linear(hidden, 2)
        self.mass_head = nn.Linear(hidden, 1)
        self.presence_head = nn.Linear(hidden, 1)
        self.horizontal_head = nn.Linear(hidden, 16)
        self.vertical_head = nn.Linear(hidden, 8)
        self.polarization_head = nn.Linear(hidden, 2)
        self.ue_head = nn.Linear(hidden, 4)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        nn.init.constant_(self.gate[-1].bias, -1.6)
        nn.init.constant_(self.presence_head.bias, -0.3)

    def forward(
        self,
        query_node: torch.Tensor,
        neighbor_node: torch.Tensor,
        edge: torch.Tensor,
        base_weights: torch.Tensor,
        component_delay: torch.Tensor,
        component_weight: torch.Tensor,
        component_valid: torch.Tensor,
        neighbor_diffuse: torch.Tensor,
        neighbor_reconstruction: torch.Tensor,
        anchor: torch.Tensor,
        fingerprint_horizontal: torch.Tensor,
        fingerprint_vertical: torch.Tensor,
        fingerprint_polarization: torch.Tensor,
        fingerprint_ue: torch.Tensor,
        fingerprint_gain_power: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, neighbors, components = component_delay.shape
        q = self.query_encoder(query_node)
        n = self.neighbor_encoder(neighbor_node)
        e = self.edge_encoder(edge)
        angle = 2.0 * math.pi * component_delay / v297.N_DELAY
        rank = torch.arange(components, device=component_delay.device).float()
        rank = rank[None, None, :].expand(batch, neighbors, -1)
        scalar = torch.stack(
            [
                torch.sin(angle), torch.cos(angle),
                torch.sin(2.0 * angle), torch.cos(2.0 * angle),
                torch.sin(4.0 * angle), torch.cos(4.0 * angle),
                torch.log(component_weight.clamp_min(1e-12)) / 12.0,
                component_weight,
                rank / max(components - 1, 1),
                component_valid.float(),
                neighbor_diffuse[:, :, None].expand(-1, -1, components),
                neighbor_reconstruction[:, :, None].expand(-1, -1, components),
                torch.log(base_weights[:, :, None].expand(-1, -1, components).clamp_min(1e-12)) / 8.0,
            ],
            dim=3,
        )
        component_feature = torch.cat(
            [
                scalar,
                fingerprint_horizontal,
                fingerprint_vertical,
                fingerprint_polarization,
                fingerprint_ue,
                fingerprint_gain_power[:, :, :, None],
            ],
            dim=3,
        )
        c = self.component_encoder(component_feature)
        q_token = q[:, None, None, :].expand(-1, neighbors, components, -1)
        n_token = n[:, :, None, :].expand(-1, -1, components, -1)
        e_token = e[:, :, None, :].expand(-1, -1, components, -1)
        token = self.token_encoder(
            torch.cat([q_token, n_token, q_token * n_token, e_token, c], dim=3)
        ).reshape(batch, neighbors * components, -1)
        mask = ~component_valid.reshape(batch, -1)
        memory = self.encoder(token, src_key_padding_mask=mask)
        slots = self.path_slots[None, :, :].expand(batch, -1, -1) + q[:, None, :]
        decoded = self.decoder(
            slots,
            memory,
            memory_key_padding_mask=mask,
        )
        delay_vector = self.delay_head(decoded)
        delay_angle = torch.atan2(delay_vector[:, :, 0], delay_vector[:, :, 1])
        delay = torch.remainder(delay_angle, 2.0 * math.pi) * v297.N_DELAY / (2.0 * math.pi)
        mass_logits = self.mass_head(decoded).squeeze(2)
        presence_logits = self.presence_head(decoded).squeeze(2)
        effective_mass = torch.softmax(
            mass_logits + F.logsigmoid(presence_logits), dim=1
        )
        horizontal = torch.softmax(self.horizontal_head(decoded), dim=2)
        vertical = torch.softmax(self.vertical_head(decoded), dim=2)
        polarization = torch.softmax(self.polarization_head(decoded), dim=2)
        ue_port = torch.softmax(self.ue_head(decoded), dim=2)

        bins = torch.arange(
            v297.N_DELAY, device=delay.device, dtype=delay.dtype
        )[None, None, :]
        delta = bins - delay[:, :, None]
        delta = torch.remainder(
            delta + v297.N_DELAY / 2.0, v297.N_DELAY
        ) - v297.N_DELAY / 2.0
        atom = torch.square(torch.sinc(delta) / torch.sinc(delta / v297.N_DELAY))
        atom /= atom.sum(dim=2, keepdim=True).clamp_min(1e-30)
        path_pdp = torch.sum(effective_mass[:, :, None] * atom, dim=1)
        pooled = decoded.mean(dim=1)
        gate = 0.65 * torch.sigmoid(self.gate(torch.cat([q, pooled], dim=1)))
        prediction = (1.0 - gate) * anchor + gate * path_pdp
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {
            "delay": delay,
            "mass": effective_mass,
            "presence_logits": presence_logits,
            "horizontal": horizontal,
            "vertical": vertical,
            "polarization": polarization,
            "ue_port": ue_port,
            "gate": gate.squeeze(1),
        }


def pair_cosine(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bsd,btd->bst", predicted, target) / (
        torch.linalg.norm(predicted, dim=2)[:, :, None]
        * torch.linalg.norm(target, dim=2)[:, None, :]
    ).clamp_min(1e-12)


def hungarian_loss(
    prediction: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    truth_pdp: torch.Tensor,
    target: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = len(prediction)
    presence_target = torch.zeros_like(diagnostics["presence_logits"])
    matched_pred = []
    matched_target = []
    with torch.no_grad():
        delay_cost = torch.abs(
            torch.remainder(
                diagnostics["delay"][:, :, None]
                - target["delay"][:, None, :]
                + v297.N_DELAY / 2.0,
                v297.N_DELAY,
            )
            - v297.N_DELAY / 2.0
        )
        fingerprint_cosine = (
            0.55 * pair_cosine(diagnostics["horizontal"], target["horizontal"])
            + 0.20 * pair_cosine(diagnostics["vertical"], target["vertical"])
            + 0.10 * pair_cosine(diagnostics["polarization"], target["polarization"])
            + 0.15 * pair_cosine(diagnostics["ue_port"], target["ue_port"])
        )
        mass_cost = torch.abs(
            torch.log(diagnostics["mass"].clamp_min(1e-8))[:, :, None]
            - torch.log(target["mass"].clamp_min(1e-8))[:, None, :]
        )
        cost = delay_cost / 0.5 + 0.6 * (1.0 - fingerprint_cosine) + 0.12 * mass_cost
        for row in range(batch):
            active = torch.nonzero(
                target["valid"][row], as_tuple=False
            ).flatten().cpu().numpy()
            if len(active) == 0:
                continue
            row_index, column_index = linear_sum_assignment(
                cost[row][:, active].detach().cpu().numpy()
            )
            target_index = active[column_index]
            presence_target[row, row_index] = 1.0
            matched_pred.extend((row, int(index)) for index in row_index)
            matched_target.extend((row, int(index)) for index in target_index)

    pdp_loss = (1.0 - F.cosine_similarity(prediction, truth_pdp, dim=1)).mean()
    root_loss = (
        1.0
        - F.cosine_similarity(
            torch.sqrt(prediction.clamp_min(1e-12)),
            torch.sqrt(truth_pdp.clamp_min(1e-12)),
            dim=1,
        )
    ).mean()
    presence_loss = F.binary_cross_entropy_with_logits(
        diagnostics["presence_logits"], presence_target
    )
    if matched_pred:
        pred_rows = torch.tensor([value[0] for value in matched_pred], device=prediction.device)
        pred_slots = torch.tensor([value[1] for value in matched_pred], device=prediction.device)
        target_rows = torch.tensor([value[0] for value in matched_target], device=prediction.device)
        target_slots = torch.tensor([value[1] for value in matched_target], device=prediction.device)
        delay_delta = torch.remainder(
            diagnostics["delay"][pred_rows, pred_slots]
            - target["delay"][target_rows, target_slots]
            + v297.N_DELAY / 2.0,
            v297.N_DELAY,
        ) - v297.N_DELAY / 2.0
        delay_loss = F.smooth_l1_loss(delay_delta, torch.zeros_like(delay_delta), beta=0.25)
        mass_loss = F.smooth_l1_loss(
            torch.log(diagnostics["mass"][pred_rows, pred_slots].clamp_min(1e-8)),
            torch.log(target["mass"][target_rows, target_slots].clamp_min(1e-8)),
            beta=0.25,
        )
        fingerprint_loss = 0.0
        for key, weight in (
            ("horizontal", 0.55),
            ("vertical", 0.20),
            ("polarization", 0.10),
            ("ue_port", 0.15),
        ):
            fingerprint_loss = fingerprint_loss + weight * (
                1.0
                - F.cosine_similarity(
                    diagnostics[key][pred_rows, pred_slots],
                    target[key][target_rows, target_slots],
                    dim=1,
                ).mean()
            )
    else:
        delay_loss = prediction.sum() * 0.0
        mass_loss = prediction.sum() * 0.0
        fingerprint_loss = prediction.sum() * 0.0
    loss = (
        pdp_loss
        + 0.10 * root_loss
        + 0.08 * delay_loss
        + 0.04 * mass_loss
        + 0.04 * fingerprint_loss
        + 0.04 * presence_loss
    )
    return loss, {
        "pdp": float(pdp_loss.detach().cpu()),
        "delay": float(delay_loss.detach().cpu()),
        "mass": float(mass_loss.detach().cpu()),
        "fingerprint": float(fingerprint_loss.detach().cpu()),
        "presence": float(presence_loss.detach().cpu()),
    }


def target_batch(
    query_ids: np.ndarray,
    rows: np.ndarray,
    component: dict[str, np.ndarray],
    fingerprint: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    ids = query_ids[rows]
    valid = component["valid"][ids].astype(bool)
    mass = component["weight"][ids].astype(np.float32) * valid
    mass /= np.maximum(mass.sum(axis=1, keepdims=True), 1e-20)
    arrays = {
        "delay": component["delay"][ids].astype(np.float32),
        "mass": mass,
        "valid": valid,
        "horizontal": fingerprint["horizontal"][ids].astype(np.float32),
        "vertical": fingerprint["vertical"][ids].astype(np.float32),
        "polarization": fingerprint["polarization"][ids].astype(np.float32),
        "ue_port": fingerprint["ue_port"][ids].astype(np.float32),
    }
    return {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}


@torch.no_grad()
def predict(
    model: HungarianPathSetTransport,
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty((len(arrays["query_ids"]), v297.N_DELAY), dtype=np.float32)
    gate = np.empty(len(output), dtype=np.float32)
    for start in range(0, len(output), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(output))
        rows = np.arange(start, end)
        batch = tensor_batch(arrays, rows, device)
        prediction, diagnostics = model(**batch)
        output[start:end] = prediction.cpu().numpy()
        gate[start:end] = diagnostics["gate"].cpu().numpy()
    return output, gate


def fit_model(
    train_arrays: dict[str, np.ndarray],
    truth_d: np.ndarray,
    component: dict[str, np.ndarray],
    fingerprint: dict[str, np.ndarray],
    seed: int,
    epochs: int,
    device: torch.device,
    validation_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[HungarianPathSetTransport, dict[str, object]]:
    set_seed(seed)
    model = HungarianPathSetTransport(
        train_arrays["query_node"].shape[1], train_arrays["edge"].shape[2]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_epoch = epochs
    best_score = -np.inf
    best_state = None
    history = []
    if validation_arrays is not None:
        validation_truth = truth_d[validation_arrays["query_ids"]]
        baseline = float(
            np.mean(v297.row_cos(validation_arrays["anchor"], validation_truth))
        )
        best_epoch = 0
        best_score = baseline
    else:
        baseline = None

    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_arrays["query_ids"]))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            batch = tensor_batch(train_arrays, rows, device)
            truth = torch.from_numpy(
                truth_d[train_arrays["query_ids"][rows]].astype(np.float32)
            ).to(device)
            target = target_batch(
                train_arrays["query_ids"], rows, component, fingerprint, device
            )
            optimizer.zero_grad(set_to_none=True)
            prediction, diagnostics = model(**batch)
            loss, parts = hungarian_loss(prediction, diagnostics, truth, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite v303 loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if validation_arrays is not None and (epoch % EVAL_EVERY == 0 or epoch == epochs):
            validation_prediction, validation_gate = predict(model, validation_arrays, device)
            score = float(np.mean(v297.row_cos(validation_prediction, validation_truth)))
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_pdp": score,
                    "validation_gate_mean": float(np.mean(validation_gate)),
                    "last_batch_parts": parts,
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
            print(
                f"v303 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "baseline_pdp": baseline,
        "best_validation_pdp": None if not np.isfinite(best_score) else float(best_score),
        "history": history,
    }


def choose_alpha(
    base: np.ndarray, model: np.ndarray, truth: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        prediction = v297.normalize_rows(base + float(alpha) * (model - base))
        rows.append(
            {
                "alpha": float(alpha),
                "pdp": float(np.mean(v297.row_cos(prediction, truth))),
            }
        )
    best = max(rows, key=lambda row: row["pdp"])
    return float(best["alpha"]), rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    component_data = np.load(v297.COMPONENT_CACHE)
    component = {key: component_data[key] for key in component_data.files}
    components = v297.component_data(component)
    fingerprint_data = np.load(v300.FINGERPRINT_CACHE)
    fingerprint = {key: fingerprint_data[key] for key in fingerprint_data.files}
    targets = np.load(v297.TARGET_CACHE)
    truth_d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    (
        all_pos,
        signatures,
        blocker_count,
        scene_features,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = v274.load_geometry()
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )

    fold_rows = []
    base_parts = []
    model_parts = []
    truth_parts = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = v273.matched_subset(
            outer_fit,
            train_pos[:, :2],
            test_pos[:, :2],
            INNER_COUNT,
            seed + 10_000,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_train_set, inner_stats, inner_node = v297.standardized_sets(
            inner_fit, inner_fit, all_pos, signatures, blocker_count, footprint,
            z_max, x_min, y_min, resolution, node_raw,
        )
        inner_val_set, _stats, _node = v297.standardized_sets(
            inner_ids, inner_fit, all_pos, signatures, blocker_count, footprint,
            z_max, x_min, y_min, resolution, node_raw, inner_stats,
        )
        inner_train = model_arrays(
            inner_train_set, inner_node, components, truth_d, fingerprint
        )
        inner_val = model_arrays(
            inner_val_set, inner_node, components, truth_d, fingerprint
        )
        _model, inner_report = fit_model(
            inner_train, truth_d, component, fingerprint, seed, EPOCHS, device, inner_val
        )
        selected_epoch = int(inner_report["best_epoch"])

        outer_train_set, outer_stats, outer_node = v297.standardized_sets(
            outer_fit, outer_fit, all_pos, signatures, blocker_count, footprint,
            z_max, x_min, y_min, resolution, node_raw,
        )
        outer_val_set, _stats, _node = v297.standardized_sets(
            outer_ids, outer_fit, all_pos, signatures, blocker_count, footprint,
            z_max, x_min, y_min, resolution, node_raw, outer_stats,
        )
        outer_train = model_arrays(
            outer_train_set, outer_node, components, truth_d, fingerprint
        )
        outer_val = model_arrays(
            outer_val_set, outer_node, components, truth_d, fingerprint
        )
        if selected_epoch == 0:
            model_prediction = outer_val["anchor"].copy()
            gate = np.zeros(len(model_prediction), dtype=np.float32)
            outer_training = {"best_epoch": 0}
        else:
            model, outer_training = fit_model(
                outer_train,
                truth_d,
                component,
                fingerprint,
                seed + 50_000,
                selected_epoch,
                device,
            )
            model_prediction, gate = predict(model, outer_val, device)
        base = outer_val["anchor"]
        truth = truth_d[outer_ids]
        base_score = float(np.mean(v297.row_cos(base, truth)))
        model_score = float(np.mean(v297.row_cos(model_prediction, truth)))
        base_parts.append(base)
        model_parts.append(model_prediction)
        truth_parts.append(truth)
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
            f"v303 fold {fold + 1}/5 epoch={selected_epoch} "
            f"base={base_score:.6f} model={model_score:.6f} delta={model_score-base_score:+.6f}",
            flush=True,
        )

    base_all = np.concatenate(base_parts)
    model_all = np.concatenate(model_parts)
    truth_all = np.concatenate(truth_parts)
    alpha, alpha_rows = choose_alpha(base_all, model_all, truth_all)
    prediction_all = v297.normalize_rows(base_all + alpha * (model_all - base_all))
    base_score = float(np.mean(v297.row_cos(base_all, truth_all)))
    score = float(np.mean(v297.row_cos(prediction_all, truth_all)))
    fold_deltas = []
    start = 0
    for row, base, model, truth in zip(fold_rows, base_parts, model_parts, truth_parts):
        end = start + len(base)
        fold_prediction = prediction_all[start:end]
        fold_score = float(np.mean(v297.row_cos(fold_prediction, truth)))
        row["blended_pdp"] = fold_score
        row["blended_delta"] = fold_score - float(row["base_pdp"])
        fold_deltas.append(float(row["blended_delta"]))
        start = end
    official_delta = score - base_score
    official_selection = bool(
        official_delta >= MIN_OFFICIAL_PDP_DELTA
        and sum(value > 0.0 for value in fold_deltas) >= 4
    )
    report = {
        "method": VERSION,
        "status": "official_selection_passed" if official_selection else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Explicit Hungarian path-set supervision resolves correspondence that "
            "aggregate PDP loss and deterministic fingerprint clustering did not."
        ),
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
        "compliance_boundary": {
            "uses": [
                "official train H continuous path-set labels",
                "official train/test planar coordinates",
                "official point-cloud scene and barrier features",
                "PyTorch Transformer encoder-decoder with Hungarian supervision",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "teacher outputs",
                "hidden labels",
                "RT coefficients or local replay truth",
            ],
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["oof"], indent=2), flush=True)


if __name__ == "__main__":
    main()
