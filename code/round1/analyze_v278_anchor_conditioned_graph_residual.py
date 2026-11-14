#!/usr/bin/env python3
"""Anchor-conditioned bounded graph residual operator for planar spectra."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v273_planar_barrier_kernel as v273
import analyze_v274_planar_barrier_attention as v274
import analyze_v277_masked_planar_graph_operator as v277
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 278
VERSION = "v278_anchor_conditioned_graph_residual"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v278_anchor_residual_targets.npz"
OUTPUT = OUT_DIR / "v278_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

LOCAL_GATE = 0.645604
OUTER_SEEDS = (278, 1278, 2278, 3278, 4278)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 100
EVAL_EVERY = 5
MIN_OFFICIAL_DELTA = 5e-4
FINAL_SEEDS = (27801, 27831, 27861)
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)


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
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def metrics(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def all_node_anchors(
    fit_ids: np.ndarray,
    all_pos: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    take = min(13, len(fit_ids))
    distance, local = cKDTree(all_pos[fit_ids, :2]).query(
        all_pos[:, :2], k=take
    )
    if take == 1:
        distance = distance[:, None]
        local = local[:, None]
    absolute = fit_ids[local]
    selected_ids = np.empty((len(all_pos), min(12, len(fit_ids) - 1)), dtype=np.int64)
    selected_distance = np.empty_like(selected_ids, dtype=np.float64)
    for node_id in range(len(all_pos)):
        keep = np.flatnonzero(absolute[node_id] != node_id)
        needed = selected_ids.shape[1]
        if len(keep) < needed:
            raise RuntimeError("insufficient leave-one-out anchor neighbors")
        keep = keep[:needed]
        selected_ids[node_id] = absolute[node_id, keep]
        selected_distance[node_id] = distance[node_id, keep]
    weight = 1.0 / np.maximum(selected_distance, 0.05) ** 2
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-20)
    sh_anchor = np.sum(
        sh[selected_ids] * weight[..., None, None], axis=1, dtype=np.float64
    )
    d_anchor = np.sum(
        d[selected_ids] * weight[..., None], axis=1, dtype=np.float64
    )
    return normalize(sh_anchor), normalize(d_anchor)


def encode_numpy(
    sh: np.ndarray,
    d: np.ndarray,
    basis: v277.SpectrumBasis,
) -> np.ndarray:
    sqrt_sh = np.sqrt(np.maximum(sh.reshape(len(sh), -1), 0.0))
    sqrt_d = np.sqrt(np.maximum(d.reshape(len(d), -1), 0.0))
    sh_coeff = (sqrt_sh - basis.sh_mean) @ basis.sh_components.T
    d_coeff = (sqrt_d - basis.d_mean) @ basis.d_components.T
    return np.concatenate(
        [
            (sh_coeff - basis.sh_coeff_mean) / basis.sh_coeff_std,
            (d_coeff - basis.d_coeff_mean) / basis.d_coeff_std,
        ],
        axis=1,
    ).astype(np.float32)


class AnchorResidualOperator(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, coeff_dim: int, sh_dim: int):
        super().__init__()
        self.sh_dim = sh_dim
        self.geometry = nn.Sequential(
            nn.Linear(node_dim, 320),
            nn.SiLU(),
            nn.LayerNorm(320),
            nn.Linear(320, v277.HIDDEN),
        )
        self.anchor = nn.Sequential(
            nn.Linear(coeff_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, v277.HIDDEN),
        )
        self.layers = nn.ModuleList(
            [
                v277.MessageLayer(v277.HIDDEN, edge_dim)
                for _ in range(v277.LAYERS)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(v277.HIDDEN),
            nn.Linear(v277.HIDDEN, 256),
            nn.SiLU(),
            nn.Linear(256, coeff_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.sh_scale = nn.Parameter(torch.tensor(-0.4, dtype=torch.float32))
        self.d_scale = nn.Parameter(torch.tensor(-0.4, dtype=torch.float32))

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        neighbors: torch.Tensor,
        anchor_coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_hidden = self.anchor(anchor_coeff)
        hidden = self.geometry(node) + anchor_hidden
        for layer in self.layers:
            hidden = layer(hidden, anchor_hidden, neighbors, edge)
        raw = torch.tanh(self.head(hidden))
        scale = torch.cat(
            [
                (0.10 + 0.45 * torch.sigmoid(self.sh_scale)).expand(self.sh_dim),
                (0.10 + 0.45 * torch.sigmoid(self.d_scale)).expand(
                    raw.shape[1] - self.sh_dim
                ),
            ]
        )
        delta = raw * scale[None, :]
        return anchor_coeff + delta, delta


@torch.no_grad()
def predict(
    model: AnchorResidualOperator,
    graph: v277.GraphData,
    anchor_coeff: np.ndarray,
    basis: v277.SpectrumBasis,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predicted, delta = model(
        torch.from_numpy(graph.node).to(device),
        torch.from_numpy(graph.edge).to(device),
        torch.from_numpy(graph.neighbors).to(device),
        torch.from_numpy(anchor_coeff).to(device),
    )
    predicted_np = predicted.cpu().numpy().astype(np.float32)
    sh, d = v277.decode_numpy(predicted_np, basis)
    return sh, d, delta.cpu().numpy().astype(np.float32)


def train_model(
    graph: v277.GraphData,
    basis: v277.SpectrumBasis,
    target_coeff: np.ndarray,
    anchor_coeff: np.ndarray,
    sh: np.ndarray,
    d: np.ndarray,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray | None,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[AnchorResidualOperator, dict[str, object]]:
    set_seed(seed)
    model = AnchorResidualOperator(
        graph.node.shape[1],
        graph.edge.shape[2],
        basis.dim,
        basis.sh_components.shape[0],
    ).to(device)
    decoder = v277.TorchDecoder(basis).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    node = torch.from_numpy(graph.node).to(device)
    edge = torch.from_numpy(graph.edge).to(device)
    neighbors = torch.from_numpy(graph.neighbors).to(device)
    anchor_tensor = torch.from_numpy(anchor_coeff).to(device)
    target_coeff_tensor = torch.from_numpy(target_coeff).to(device)
    sh_flat = torch.from_numpy(sh.reshape(len(sh), -1)).to(device)
    d_flat = torch.from_numpy(d.reshape(len(d), -1)).to(device)
    fit = torch.from_numpy(fit_ids).to(device)
    baseline_sh, baseline_d = v277.decode_numpy(anchor_coeff, basis)
    baseline_metric = None
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    if validation_ids is not None:
        baseline_metric = metrics(
            baseline_sh[validation_ids],
            baseline_d[validation_ids],
            sh[validation_ids],
            d[validation_ids],
        )
        best_metric = baseline_metric["weighted"]
        best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        predicted, delta = model(node, edge, neighbors, anchor_tensor)
        pred_coeff = predicted[fit]
        pred_sh, pred_d = decoder(pred_coeff)
        sh_loss = 1.0 - F.cosine_similarity(pred_sh, sh_flat[fit], dim=1).mean()
        d_loss = 1.0 - F.cosine_similarity(pred_d, d_flat[fit], dim=1).mean()
        coeff_loss = F.smooth_l1_loss(
            pred_coeff, target_coeff_tensor[fit], beta=0.8
        )
        residual_penalty = delta[fit].square().mean()
        loss = (
            0.49 * sh_loss
            + 0.45 * d_loss
            + 0.04 * coeff_loss
            + 0.02 * residual_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs:
            item: dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
            }
            if validation_ids is not None:
                pred_val_sh, pred_val_d, pred_delta = predict(
                    model, graph, anchor_coeff, basis, device
                )
                val_metric = metrics(
                    pred_val_sh[validation_ids],
                    pred_val_d[validation_ids],
                    sh[validation_ids],
                    d[validation_ids],
                )
                item["validation"] = val_metric
                item["baseline"] = baseline_metric
                item["delta_rms"] = float(
                    np.sqrt(np.mean(pred_delta[validation_ids] ** 2))
                )
                if val_metric["weighted"] > best_metric:
                    best_metric = val_metric["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"v278 seed={seed} epoch={epoch}/{epochs} loss={float(loss.detach().cpu()):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": float(best_metric) if np.isfinite(best_metric) else None,
        "baseline": baseline_metric,
        "history": history,
    }


def choose_residual_alpha(
    base: np.ndarray, model: np.ndarray, truth: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        blended = normalize(base + alpha * (model - base))
        rows.append(
            {
                "alpha": float(alpha),
                "cosine": float(np.mean(row_cos(blended, truth))),
            }
        )
    best = max(rows, key=lambda row: row["cosine"])
    return float(best["alpha"]), rows


def main() -> None:
    for path in (
        BASE,
        TARGET_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = np.load(TARGET_CACHE)
    sh = targets["sh"].astype(np.float32)
    d = targets["d"].astype(np.float32)
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
    all_ids = np.arange(len(all_pos), dtype=np.int64)
    raw_graph = v274.build_raw_set(
        all_ids,
        all_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )

    outer_rows: list[dict[str, object]] = []
    outer_base_sh: list[np.ndarray] = []
    outer_base_d: list[np.ndarray] = []
    outer_model_sh: list[np.ndarray] = []
    outer_model_d: list[np.ndarray] = []
    outer_truth_sh: list[np.ndarray] = []
    outer_truth_d: list[np.ndarray] = []
    selected_epochs: list[int] = []
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

        inner_basis, inner_target_coeff, inner_basis_report = v277.fit_basis(
            sh, d, inner_fit
        )
        inner_anchor_sh, inner_anchor_d = all_node_anchors(
            inner_fit, all_pos, sh, d
        )
        inner_anchor_coeff = encode_numpy(
            inner_anchor_sh, inner_anchor_d, inner_basis
        )
        inner_graph = v277.graph_data(node_raw, raw_graph, inner_fit)
        _inner_model, inner_training = train_model(
            inner_graph,
            inner_basis,
            inner_target_coeff,
            inner_anchor_coeff,
            sh,
            d,
            inner_fit,
            inner_ids,
            seed,
            EPOCHS,
            device,
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)

        outer_basis, outer_target_coeff, outer_basis_report = v277.fit_basis(
            sh, d, outer_fit
        )
        outer_anchor_sh, outer_anchor_d = all_node_anchors(
            outer_fit, all_pos, sh, d
        )
        outer_anchor_coeff = encode_numpy(
            outer_anchor_sh, outer_anchor_d, outer_basis
        )
        outer_graph = v277.graph_data(node_raw, raw_graph, outer_fit)
        if selected_epoch == 0:
            pred_sh, pred_d = outer_anchor_sh, outer_anchor_d
            outer_training = {"seed": seed + 50_000, "best_epoch": 0, "history": []}
            pred_delta = np.zeros_like(outer_anchor_coeff)
        else:
            outer_model, outer_training = train_model(
                outer_graph,
                outer_basis,
                outer_target_coeff,
                outer_anchor_coeff,
                sh,
                d,
                outer_fit,
                None,
                seed + 50_000,
                selected_epoch,
                device,
            )
            pred_sh, pred_d, pred_delta = predict(
                outer_model,
                outer_graph,
                outer_anchor_coeff,
                outer_basis,
                device,
            )
        baseline = metrics(
            outer_anchor_sh[outer_ids],
            outer_anchor_d[outer_ids],
            sh[outer_ids],
            d[outer_ids],
        )
        model_metric = metrics(
            pred_sh[outer_ids], pred_d[outer_ids], sh[outer_ids], d[outer_ids]
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
                "inner_basis": inner_basis_report,
                "inner_training": inner_training,
                "outer_basis": outer_basis_report,
                "outer_training": outer_training,
                "baseline": baseline,
                "model": model_metric,
                "delta": delta,
                "residual_rms": summarize(
                    np.sqrt(np.mean(pred_delta[outer_ids] ** 2, axis=1))
                ),
            }
        )
        outer_base_sh.append(outer_anchor_sh[outer_ids])
        outer_base_d.append(outer_anchor_d[outer_ids])
        outer_model_sh.append(pred_sh[outer_ids])
        outer_model_d.append(pred_d[outer_ids])
        outer_truth_sh.append(sh[outer_ids])
        outer_truth_d.append(d[outer_ids])
        print(
            f"v278 outer {fold + 1}/{len(OUTER_SEEDS)} epoch={selected_epoch} "
            f"baseline={baseline['weighted']:.6f} model={model_metric['weighted']:.6f} "
            f"delta={delta['weighted']:+.6f}",
            flush=True,
        )

    fold_delta = np.asarray(
        [row["delta"]["weighted"] for row in outer_rows], dtype=np.float64
    )
    fold_sh_delta = np.asarray(
        [row["delta"]["sh"] for row in outer_rows], dtype=np.float64
    )
    fold_d_delta = np.asarray(
        [row["delta"]["d"] for row in outer_rows], dtype=np.float64
    )
    official_selected = bool(
        np.mean(fold_delta) >= MIN_OFFICIAL_DELTA
        and np.sum(fold_delta > 0.0) >= 3
        and np.mean(fold_sh_delta) > 0.0
        and np.mean(fold_d_delta) > 0.0
    )
    cv_base_sh = np.concatenate(outer_base_sh)
    cv_base_d = np.concatenate(outer_base_d)
    cv_model_sh = np.concatenate(outer_model_sh)
    cv_model_d = np.concatenate(outer_model_d)
    cv_truth_sh = np.concatenate(outer_truth_sh)
    cv_truth_d = np.concatenate(outer_truth_d)
    alpha_sh, alpha_sh_grid = choose_residual_alpha(
        cv_base_sh, cv_model_sh, cv_truth_sh
    )
    alpha_d, alpha_d_grid = choose_residual_alpha(
        cv_base_d, cv_model_d, cv_truth_d
    )

    final_training: list[dict[str, object]] = []
    local_score = None
    projection_diagnostics = None
    if official_selected:
        positive_epochs = [epoch for epoch in selected_epochs if epoch > 0]
        final_epoch = max(int(np.median(positive_epochs)), 1)
        full_basis, full_target_coeff, full_basis_report = v277.fit_basis(
            sh, d, valid_ids
        )
        full_anchor_sh, full_anchor_d = all_node_anchors(
            valid_ids, all_pos, sh, d
        )
        full_anchor_coeff = encode_numpy(
            full_anchor_sh, full_anchor_d, full_basis
        )
        full_graph = v277.graph_data(node_raw, raw_graph, valid_ids)
        test_sh_parts = []
        test_d_parts = []
        for seed in FINAL_SEEDS:
            model, training = train_model(
                full_graph,
                full_basis,
                full_target_coeff,
                full_anchor_coeff,
                sh,
                d,
                valid_ids,
                None,
                seed,
                final_epoch,
                device,
            )
            pred_sh, pred_d, _ = predict(
                model,
                full_graph,
                full_anchor_coeff,
                full_basis,
                device,
            )
            test_sh_parts.append(pred_sh[len(train_pos) :])
            test_d_parts.append(pred_d[len(train_pos) :])
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "model_state": model.state_dict(),
                    "basis": full_basis,
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
        test_model_sh = normalize(np.mean(test_sh_parts, axis=0))
        test_model_d = normalize(np.mean(test_d_parts, axis=0))
        test_base_sh = full_anchor_sh[len(train_pos) :]
        test_base_d = full_anchor_d[len(train_pos) :]
        current_sh, current_d = v274.extract_current_marginals(BASE)
        target_sh = normalize(
            current_sh + alpha_sh * (test_model_sh - test_base_sh)
        )
        target_d = normalize(
            current_d + alpha_d * (test_model_d - test_base_d)
        )
        np.savez_compressed(
            TARGETS_OUT,
            base_sh=test_base_sh,
            base_d=test_base_d,
            model_sh=test_model_sh,
            model_d=test_model_d,
            target_sh=target_sh,
            target_d=target_d,
            alpha_sh=np.asarray(alpha_sh, dtype=np.float32),
            alpha_d=np.asarray(alpha_d, dtype=np.float32),
            basis_report_json=np.asarray(json.dumps(full_basis_report)),
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
            "The weak positive component direction found by v277 becomes usable when the "
            "graph network predicts only a bounded residual around a strict LOO KNN anchor."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP labels",
                "official train/test XY coordinates",
                "official point-cloud wall/component/height features",
                "strict v239 official-only carrier",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher outputs",
                "hidden labels",
                "RT MIMO H or path coefficients",
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
            "epochs": EPOCHS,
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
            "final_seeds": list(FINAL_SEEDS),
        },
        "outer_folds": outer_rows,
        "official_selection": {
            "mean_delta": {
                "sh": float(np.mean(fold_sh_delta)),
                "d": float(np.mean(fold_d_delta)),
                "weighted": float(np.mean(fold_delta)),
            },
            "weighted_wins": int(np.sum(fold_delta > 0.0)),
            "selected_epochs": selected_epochs,
            "selected": official_selected,
            "residual_alpha_sh": alpha_sh,
            "residual_alpha_d": alpha_d,
            "alpha_sh_grid": alpha_sh_grid,
            "alpha_d_grid": alpha_d_grid,
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
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
