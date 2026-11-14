#!/usr/bin/env python3
"""Path-family component transport with learned delay-bin quantization residuals."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
import analyze_v289_path_conditioned_component_transport as v289
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 290
VERSION = "v290_subbin_path_component_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v290_subbin_transport_targets.npz"
OUTPUT = OUT_DIR / "v290_Round1_Test_Channel.npy"

LOCAL_GATE = 0.645604
PATH_FAMILIES = 2
S_RESIDUALS = (-1, 0, 1)
OPTIONS = 1 + PATH_FAMILIES * len(S_RESIDUALS)
OPTION_DIM = 16
EPOCHS = 60
EVAL_EVERY = 5
BATCH_SIZE = 40
OUTER_SEEDS = (290, 1290, 2290, 3290, 4290)
OUTER_COUNT = 320
INNER_COUNT = 240
FINAL_SEEDS = (29001, 29031, 29061)
MIN_OFFICIAL_DELTA = 5e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expand_subbin_options(base: v289.TransportSet) -> v289.TransportSet:
    count, tokens = base.source_index.shape
    destination = np.repeat(
        base.source_index[:, :, None], OPTIONS, axis=2
    ).astype(np.int64)
    patch_id = np.zeros((count, tokens, OPTIONS), dtype=np.int64)
    numeric = np.zeros((count, tokens, OPTIONS, OPTION_DIM), dtype=np.float32)
    prior = np.full((count, tokens, OPTIONS), -6.0, dtype=np.float32)
    valid = np.zeros((count, tokens, OPTIONS), dtype=bool)
    valid[:, :, 0] = True
    prior[:, :, 0] = base.option_prior[:, :, 0]
    numeric[:, :, 0, :14] = base.option_numeric[:, :, 0]

    output_option = 1
    for family in range(1, PATH_FAMILIES + 1):
        family_valid = base.option_valid[:, :, family]
        base_destination = base.option_destination[:, :, family]
        base_s = base_destination // v289.H_SIZE
        base_h = base_destination % v289.H_SIZE
        physical_delta = 4.0 * base.option_numeric[:, :, family, 3]
        fractional = physical_delta - np.rint(physical_delta)
        for residual in S_RESIDUALS:
            destination[:, :, output_option] = (
                np.mod(base_s + residual, v289.S_SIZE) * v289.H_SIZE + base_h
            )
            patch_id[:, :, output_option] = base.option_patch_id[:, :, family]
            numeric[:, :, output_option, :14] = base.option_numeric[:, :, family]
            numeric[:, :, output_option, 14] = float(residual)
            numeric[:, :, output_option, 15] = (residual * fractional).astype(
                np.float32
            )
            prior[:, :, output_option] = (
                base.option_prior[:, :, family]
                - 0.62 * abs(residual)
                + 0.28 * residual * fractional
            )
            valid[:, :, output_option] = family_valid
            output_option += 1

    return v289.TransportSet(
        query_ids=base.query_ids,
        neighbor_ids=base.neighbor_ids,
        query_node=base.query_node,
        neighbor_node=base.neighbor_node,
        anchor=base.anchor,
        source_index=base.source_index,
        source_mass=base.source_mass,
        token_numeric=base.token_numeric,
        option_destination=destination,
        option_patch_id=patch_id,
        option_numeric=numeric,
        option_prior=prior,
        option_valid=valid,
        shared_path_count=base.shared_path_count,
    )


def build_transport_set(*args, **kwargs) -> v289.TransportSet:
    return expand_subbin_options(v289.build_transport_set(*args, **kwargs))


class SubbinPathTransportNet(nn.Module):
    def __init__(self, node_dim: int, patch_count: int) -> None:
        super().__init__()
        self.node = nn.Sequential(
            nn.Linear(node_dim, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 80),
        )
        self.token = nn.Sequential(
            nn.Linear(12, 80), nn.SiLU(), nn.Linear(80, 48)
        )
        self.option = nn.Sequential(
            nn.Linear(OPTION_DIM, 80), nn.SiLU(), nn.Linear(80, 48)
        )
        self.patch = nn.Embedding(patch_count + 2, 24)
        self.trunk = nn.Sequential(
            nn.Linear(80 * 3 + 48 * 2 + 24, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 160),
            nn.SiLU(),
            nn.Linear(160, 1),
        )

    def forward(
        self,
        query_node: torch.Tensor,
        neighbor_node: torch.Tensor,
        anchor: torch.Tensor,
        source_index: torch.Tensor,
        source_mass: torch.Tensor,
        token_numeric: torch.Tensor,
        option_destination: torch.Tensor,
        option_patch_id: torch.Tensor,
        option_numeric: torch.Tensor,
        option_prior: torch.Tensor,
        option_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, options = option_destination.shape
        query = self.node(query_node)[:, None, None, :].expand(-1, tokens, options, -1)
        donor = self.node(neighbor_node)
        donor = donor[:, :, None, :].expand(-1, -1, v289.PEAKS, -1)
        donor = donor.reshape(batch, tokens, -1)[:, :, None, :].expand(-1, -1, options, -1)
        token = self.token(token_numeric)[:, :, None, :].expand(-1, -1, options, -1)
        option = self.option(option_numeric)
        hidden = torch.cat(
            [query, donor, query * donor, token, option, self.patch(option_patch_id)],
            dim=3,
        )
        delta = 2.4 * torch.tanh(self.trunk(hidden).squeeze(3))
        logits = (option_prior + delta).masked_fill(~option_valid, -1e9)
        probability = torch.softmax(logits, dim=2)

        removed = torch.zeros_like(anchor)
        removed.scatter_add_(1, source_index, source_mass)
        added = torch.zeros_like(anchor)
        added.scatter_add_(
            1,
            option_destination.reshape(batch, -1),
            (source_mass[:, :, None] * probability).reshape(batch, -1),
        )
        prediction = torch.clamp(anchor - removed + added, min=0.0)
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, probability


def batch_forward(
    model: SubbinPathTransportNet,
    data: v289.TransportSet,
    rows: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        torch.from_numpy(data.query_node[rows]).to(device),
        torch.from_numpy(data.neighbor_node[rows]).to(device),
        torch.from_numpy(data.anchor[rows].reshape(len(rows), -1)).to(device),
        torch.from_numpy(data.source_index[rows]).to(device),
        torch.from_numpy(data.source_mass[rows]).to(device),
        torch.from_numpy(data.token_numeric[rows]).to(device),
        torch.from_numpy(data.option_destination[rows]).to(device),
        torch.from_numpy(data.option_patch_id[rows]).to(device),
        torch.from_numpy(data.option_numeric[rows]).to(device),
        torch.from_numpy(data.option_prior[rows]).to(device),
        torch.from_numpy(data.option_valid[rows]).to(device),
    )


@torch.no_grad()
def predict(
    model: SubbinPathTransportNet,
    data: v289.TransportSet,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty(
        (len(data.query_ids), v289.S_SIZE, v289.H_SIZE), dtype=np.float32
    )
    move = np.empty(len(data.query_ids), dtype=np.float32)
    s_move = np.empty(len(data.query_ids), dtype=np.float32)
    residual_values = np.asarray(
        [0] + [value for _ in range(PATH_FAMILIES) for value in S_RESIDUALS],
        dtype=np.float32,
    )
    for start in range(0, len(data.query_ids), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(data.query_ids))
        rows = np.arange(start, end, dtype=np.int64)
        value, probability = batch_forward(model, data, rows, device)
        output[start:end] = value.reshape(
            -1, v289.S_SIZE, v289.H_SIZE
        ).cpu().numpy()
        move[start:end] = (1.0 - probability[:, :, 0]).mean(dim=1).cpu().numpy()
        residual = torch.from_numpy(residual_values).to(device)
        s_move[start:end] = torch.sum(
            probability * residual[None, None, :], dim=2
        ).abs().mean(dim=1).cpu().numpy()
    return v279.normalize(output), move, s_move


def train_model(
    train_data: v289.TransportSet,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    patch_count: int,
    validation: v289.TransportSet | None,
) -> tuple[SubbinPathTransportNet, dict[str, object]]:
    set_seed(seed)
    model = SubbinPathTransportNet(
        train_data.query_node.shape[1], patch_count
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    validation_base = None
    if validation is not None:
        validation_base = v279.metrics(
            validation.anchor,
            truth_sh[validation.query_ids],
            truth_d[validation.query_ids],
        )
        best_metric = validation_base["weighted"]
        best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_data.query_ids))
        losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            prediction, probability = batch_forward(model, train_data, rows, device)
            target_sh = torch.from_numpy(
                truth_sh[train_data.query_ids[rows]].reshape(len(rows), -1)
            ).to(device)
            target_d = torch.from_numpy(truth_d[train_data.query_ids[rows]]).to(device)
            pred_d = v279.derive_d_torch(prediction)
            sh_loss = 1.0 - F.cosine_similarity(prediction, target_sh, dim=1).mean()
            d_loss = 1.0 - F.cosine_similarity(pred_d, target_d, dim=1).mean()
            move = (1.0 - probability[:, :, 0]).mean()
            loss = 0.52 * sh_loss + 0.48 * d_loss + 8e-5 * move
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
                value, move, s_move = predict(model, validation, device)
                metric = v279.metrics(
                    value,
                    truth_sh[validation.query_ids],
                    truth_d[validation.query_ids],
                )
                item["validation"] = metric
                item["baseline"] = validation_base
                item["move"] = summarize(move)
                item["s_move"] = summarize(s_move)
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
                f"v290 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": float(best_metric) if np.isfinite(best_metric) else None,
        "baseline": validation_base,
        "history": history,
    }


def main() -> None:
    for path in (
        v289.BASE,
        v289.TARGET_CACHE,
        v289.TRAIN_CANDIDATES,
        v289.TEST_CANDIDATES,
        v289.SCENE_CACHE,
        v289.V265_REPORT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_data = np.load(v289.TARGET_CACHE)
    truth_sh = v279.normalize(target_data["sh"].astype(np.float32))
    truth_d = v279.normalize(target_data["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    train_node, test_node = v257.build_node_features(
        train_pos[:, :2], test_pos[:, :2]
    )
    patches = np.load(v289.SCENE_CACHE)["wall_patches"].astype(np.float32)
    train_candidates = dict(np.load(v289.TRAIN_CANDIDATES))
    test_candidates = dict(np.load(v289.TEST_CANDIDATES))
    with v289.V265_REPORT.open("r", encoding="utf-8") as handle:
        best_spec = json.load(handle)["selection"]["best"]
    train_geometry = v289.build_path_geometry(
        train_pos, train_candidates, train_node, patches, best_spec
    )
    test_geometry = v289.build_path_geometry(
        test_pos, test_candidates, test_node, patches, best_spec
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v290 device={device} valid={len(valid_ids)}", flush=True)

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

        mean, std = v289.node_stats(train_node, inner_fit)
        inner_train = build_transport_set(
            inner_fit,
            inner_fit,
            train_pos,
            train_pos,
            train_node,
            train_node,
            train_geometry,
            train_geometry,
            truth_sh,
            True,
        ).standardize_nodes(mean, std)
        inner_validation = build_transport_set(
            inner_ids,
            inner_fit,
            train_pos,
            train_pos,
            train_node,
            train_node,
            train_geometry,
            train_geometry,
            truth_sh,
            False,
        ).standardize_nodes(mean, std)
        _, inner_training = train_model(
            inner_train,
            truth_sh,
            truth_d,
            seed,
            EPOCHS,
            device,
            len(patches),
            inner_validation,
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)

        mean, std = v289.node_stats(train_node, outer_fit)
        outer_train = build_transport_set(
            outer_fit,
            outer_fit,
            train_pos,
            train_pos,
            train_node,
            train_node,
            train_geometry,
            train_geometry,
            truth_sh,
            True,
        ).standardize_nodes(mean, std)
        outer_validation = build_transport_set(
            outer_ids,
            outer_fit,
            train_pos,
            train_pos,
            train_node,
            train_node,
            train_geometry,
            train_geometry,
            truth_sh,
            False,
        ).standardize_nodes(mean, std)
        if selected_epoch == 0:
            model_sh = outer_validation.anchor.copy()
            move = np.zeros(len(outer_ids), dtype=np.float32)
            s_move = np.zeros(len(outer_ids), dtype=np.float32)
            outer_training = {"seed": seed + 100_000, "best_epoch": 0, "history": []}
        else:
            model, outer_training = train_model(
                outer_train,
                truth_sh,
                truth_d,
                seed + 100_000,
                selected_epoch,
                device,
                len(patches),
                None,
            )
            model_sh, move, s_move = predict(model, outer_validation, device)
        baseline = v279.metrics(
            outer_validation.anchor, truth_sh[outer_ids], truth_d[outer_ids]
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
                "move": summarize(move),
                "s_move": summarize(s_move),
            }
        )
        oof_ids_parts.append(outer_ids)
        oof_base_parts.append(outer_validation.anchor)
        oof_model_parts.append(model_sh)
        print(
            f"v290 outer {fold + 1}/{len(OUTER_SEEDS)} epoch={selected_epoch} "
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
    weighted_wins = int(np.sum([value["weighted"] > 0.0 for value in fold_delta]))
    official_selected = bool(
        alpha > 0.0
        and mean_delta["weighted"] >= MIN_OFFICIAL_DELTA
        and weighted_wins >= 3
        and mean_delta["sh"] > 0.0
        and mean_delta["d"] > 0.0
    )

    final_training: list[dict[str, object]] = []
    projection_diagnostics = None
    local_score = None
    test_move = None
    test_s_move = None
    if official_selected:
        positive_epochs = [value for value in selected_epochs if value > 0]
        final_epoch = max(int(np.median(positive_epochs)), 1)
        mean, std = v289.node_stats(train_node, valid_ids)
        final_train = build_transport_set(
            valid_ids,
            valid_ids,
            train_pos,
            train_pos,
            train_node,
            train_node,
            train_geometry,
            train_geometry,
            truth_sh,
            True,
        ).standardize_nodes(mean, std)
        test_ids = np.arange(len(test_pos), dtype=np.int64)
        final_test = build_transport_set(
            test_ids,
            valid_ids,
            test_pos,
            train_pos,
            test_node,
            train_node,
            test_geometry,
            train_geometry,
            truth_sh,
            False,
        ).standardize_nodes(mean, std)
        test_parts: list[np.ndarray] = []
        move_parts: list[np.ndarray] = []
        s_move_parts: list[np.ndarray] = []
        for seed in FINAL_SEEDS:
            model, training = train_model(
                final_train,
                truth_sh,
                truth_d,
                seed,
                final_epoch,
                device,
                len(patches),
                None,
            )
            test_model, move, s_move = predict(model, final_test, device)
            test_parts.append(test_model)
            move_parts.append(move)
            s_move_parts.append(s_move)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "model_state": model.state_dict(),
                    "node_mean": mean,
                    "node_std": std,
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
        test_model = v279.normalize(np.mean(test_parts, axis=0))
        test_move = np.mean(move_parts, axis=0).astype(np.float32)
        test_s_move = np.mean(s_move_parts, axis=0).astype(np.float32)
        current_sh, _ = v274.extract_current_marginals(v289.BASE)
        residual = test_model - final_test.anchor
        target_sh = v279.normalize(current_sh + alpha * residual)
        target_d = v279.derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            anchor_sh=final_test.anchor,
            model_sh=test_model,
            current_sh=current_sh,
            target_sh=target_sh,
            target_d=target_d,
            residual=residual,
            alpha=np.asarray(alpha, dtype=np.float32),
            move=test_move,
            s_move=test_s_move,
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
            "v289 preserved wall-path identity but rounded sub-bin physical delay changes to zero. "
            "A learned -1/0/+1 S-bin residual within the nearest physical path families should "
            "recover the quantized delay jumps seen by the v255 transport oracle."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP labels",
                "official train/test planar coordinates",
                "official point-cloud wall reflection families",
                "PyTorch sub-bin option network trained only on official train folds",
                "strict v239 official-only carrier after official selection",
            ],
            "does_not_use": [
                "synthetic/Sionna channel labels",
                "v212/v217 teacher output",
                "hidden labels or RT H/path coefficients",
                "local replay truth for training, epoch, alpha, or option selection",
            ],
            "local_truth_usage": (
                "one final gate score after nested official selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "neighbors": v289.NEIGHBORS,
            "peaks_per_neighbor": v289.PEAKS,
            "path_families": PATH_FAMILIES,
            "s_residuals": list(S_RESIDUALS),
            "option_count": OPTIONS,
            "epochs": EPOCHS,
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "final_seeds": list(FINAL_SEEDS),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
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
        "test_move": summarize(test_move) if test_move is not None else None,
        "test_s_move": summarize(test_s_move) if test_s_move is not None else None,
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
