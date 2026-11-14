#!/usr/bin/env python3
"""Joint PDP and S-H renderer from identity-matched official wall paths."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v254_route_topology_spectral_field as v254
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v266_learned_scatterer_joint_projection as v266
import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v306_matched_patch_energy_transformer as v306
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 308
VERSION = "v308_joint_matched_patch_renderer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v308_joint_targets.npz"
OUTPUT = OUT_DIR / "v308_Round1_Test_Channel.npy"

OUTER_SEEDS = (308, 1308, 2308, 3308, 4308)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 80
EVAL_EVERY = 5
BATCH_SIZE = 64
K_ANCHOR = 16
ALPHA_GRID = np.linspace(0.0, 1.25, 11, dtype=np.float32)
MIN_OFFICIAL_JOINT_DELTA = 0.01
FINAL_SEEDS = (30801, 30831, 30861)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def candidate_h_center(
    pos: np.ndarray, candidates: dict[str, np.ndarray]
) -> np.ndarray:
    count = len(pos)
    center = np.zeros((count, v306.PATH_COUNT), dtype=np.float32)
    direct_vector = pos[:, :2] - v304.BS[:2]
    direct_azimuth = np.arctan2(direct_vector[:, 1], direct_vector[:, 0])
    center[:, 0] = v304.expected_h_bin(direct_azimuth)
    point = candidates["reflection_point"][:, : v306.REFLECTIONS]
    vector = point[..., :2] - v304.BS[:2]
    azimuth = np.arctan2(vector[..., 1], vector[..., 0])
    center[:, 1:] = v304.expected_h_bin(azimuth)
    return center


def exact_atoms(center: torch.Tensor, size: int) -> torch.Tensor:
    bins = torch.arange(size, device=center.device, dtype=center.dtype)[None, None, :]
    delta = bins - center[:, :, None]
    delta = torch.remainder(delta + size / 2.0, size) - size / 2.0
    atom = torch.square(torch.sinc(delta) / torch.sinc(delta / size))
    atom /= atom.sum(dim=2, keepdim=True).clamp_min(1e-30)
    return atom


def render_sh(
    weights: torch.Tensor,
    delay: torch.Tensor,
    h_center: torch.Tensor,
) -> torch.Tensor:
    s_center = torch.remainder(-delay, v297.N_DELAY)
    s_atom = exact_atoms(s_center, v297.N_DELAY)
    h_atom = exact_atoms(h_center, 16)
    output = torch.einsum("bp,bps,bph->bsh", weights, s_atom, h_atom)
    output /= output.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
    return output


def generic_knn_anchor(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    train_pos: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    tree = cKDTree(train_pos[bank_ids, :2])
    count = min(K_ANCHOR + 1, len(bank_ids))
    distance, local = tree.query(train_pos[query_ids, :2], k=count)
    if count == 1:
        distance = distance[:, None]
        local = local[:, None]
    bank_lookup = bank_ids[local]
    selected_count = min(K_ANCHOR, len(bank_ids))
    selected = np.empty((len(query_ids), selected_count), dtype=np.int64)
    selected_distance = np.empty((len(query_ids), selected_count), dtype=np.float64)
    for row, query_id in enumerate(query_ids):
        keep = bank_lookup[row] != query_id
        ids = bank_lookup[row, keep][:selected_count]
        dist = distance[row, keep][:selected_count]
        if len(ids) < selected_count:
            ids = bank_lookup[row, :selected_count]
            dist = distance[row, :selected_count]
        selected[row] = ids
        selected_distance[row] = dist
    weight = 1.0 / np.maximum(selected_distance, 0.05) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    reshape = (len(query_ids), selected_count) + (1,) * (truth.ndim - 1)
    prediction = np.sum(
        weight.reshape(reshape) * truth[selected], axis=1, dtype=np.float64
    )
    return v297.normalize_rows(prediction)


@torch.no_grad()
def predict_joint(
    model: v306.MatchedPatchEnergyTransformer,
    feature: v264.FeatureSet,
    h_center: np.ndarray,
    anchor_d: np.ndarray,
    anchor_sh: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output_d = np.empty_like(anchor_d)
    output_sh = np.empty_like(anchor_sh)
    gate_out = np.empty(len(anchor_d), dtype=np.float32)
    for start in range(0, len(anchor_d), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(anchor_d))
        ids = np.arange(start, end)
        anchor_tensor = torch.from_numpy(anchor_d[ids]).to(device)
        prediction_d, diagnostics = model(feature, ids, anchor_tensor, device)
        delay = torch.from_numpy(feature.delay[ids]).to(device)
        h = torch.from_numpy(h_center[ids]).to(device)
        path_sh = render_sh(diagnostics["weights"], delay, h)
        gate = diagnostics["gate"][:, None, None]
        prediction_sh = (1.0 - gate) * torch.from_numpy(anchor_sh[ids]).to(device) + gate * path_sh
        prediction_sh /= prediction_sh.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        output_d[start:end] = prediction_d.cpu().numpy()
        output_sh[start:end] = prediction_sh.cpu().numpy()
        gate_out[start:end] = diagnostics["gate"].cpu().numpy()
    return output_sh, output_d, gate_out


def fit_model(
    feature: v264.FeatureSet,
    h_center: np.ndarray,
    anchor_sh: np.ndarray,
    anchor_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    seed: int,
    epochs: int,
    patch_count: int,
    device: torch.device,
    validation: tuple[v264.FeatureSet, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[v306.MatchedPatchEnergyTransformer, dict[str, object]]:
    set_seed(seed)
    model = v306.MatchedPatchEnergyTransformer(
        feature.global_features.shape[1], feature.path_numeric.shape[2], patch_count
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_epoch = epochs
    best_score = -np.inf
    best_state = None
    history = []
    if validation is not None:
        val_feature, val_h, val_anchor_sh, val_anchor_d, val_truth_sh, val_truth_d = validation
        baseline_sh = float(np.mean(v297.row_cos(val_anchor_sh, val_truth_sh)))
        baseline_d = float(np.mean(v297.row_cos(val_anchor_d, val_truth_d)))
        best_score = 0.52 * baseline_sh + 0.48 * baseline_d
        best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(truth_d))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            anchor_d_tensor = torch.from_numpy(anchor_d[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction_d, diagnostics = model(feature, ids, anchor_d_tensor, device)
            delay = torch.from_numpy(feature.delay[ids]).to(device)
            h = torch.from_numpy(h_center[ids]).to(device)
            path_sh = render_sh(diagnostics["weights"], delay, h)
            gate = diagnostics["gate"][:, None, None]
            prediction_sh = (1.0 - gate) * torch.from_numpy(anchor_sh[ids]).to(device) + gate * path_sh
            prediction_sh /= prediction_sh.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
            target_d = torch.from_numpy(truth_d[ids]).to(device)
            target_sh = torch.from_numpy(truth_sh[ids]).to(device)
            loss_d = (1.0 - F.cosine_similarity(prediction_d, target_d, dim=1)).mean()
            loss_sh = (
                1.0
                - F.cosine_similarity(
                    prediction_sh.reshape(len(ids), -1),
                    target_sh.reshape(len(ids), -1),
                    dim=1,
                )
            ).mean()
            label_loss = v306.default_supervision_loss(
                diagnostics,
                torch.from_numpy(labels[ids]).to(device),
                torch.from_numpy(quality[ids]).to(device),
            )
            loss = 0.52 * loss_sh + 0.48 * loss_d + 0.08 * label_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite v308 loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if validation is not None and (epoch % EVAL_EVERY == 0 or epoch == epochs):
            pred_sh, pred_d, gate = predict_joint(
                model, val_feature, val_h, val_anchor_d, val_anchor_sh, device
            )
            sh_score = float(np.mean(v297.row_cos(pred_sh, val_truth_sh)))
            d_score = float(np.mean(v297.row_cos(pred_d, val_truth_d)))
            weighted = 0.52 * sh_score + 0.48 * d_score
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_sh": sh_score,
                    "validation_d": d_score,
                    "validation_weighted": weighted,
                    "validation_gate_mean": float(np.mean(gate)),
                }
            )
            if weighted > best_score:
                best_epoch = epoch
                best_score = weighted
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if epoch % 20 == 0 or epoch == epochs:
            print(f"v308 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_validation_weighted": None if not np.isfinite(best_score) else float(best_score),
        "history": history,
    }


def choose_alphas(
    base_sh: np.ndarray,
    model_sh: np.ndarray,
    truth_sh: np.ndarray,
    base_d: np.ndarray,
    model_d: np.ndarray,
    truth_d: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = []
    for alpha_sh in ALPHA_GRID:
        sh = v297.normalize_rows(base_sh + float(alpha_sh) * (model_sh - base_sh))
        sh_score = float(np.mean(v297.row_cos(sh, truth_sh)))
        for alpha_d in ALPHA_GRID:
            d = v297.normalize_rows(base_d + float(alpha_d) * (model_d - base_d))
            d_score = float(np.mean(v297.row_cos(d, truth_d)))
            rows.append(
                {
                    "alpha_sh": float(alpha_sh),
                    "alpha_d": float(alpha_d),
                    "sh": sh_score,
                    "d": d_score,
                    "weighted": 0.52 * sh_score + 0.48 * d_score,
                }
            )
    best = max(rows, key=lambda row: row["weighted"])
    return best, rows


def main() -> None:
    for path in (v306.TRAIN_CANDIDATES, v306.TARGET_CACHE, v306.BASE, v306.SCENE_CACHE, v306.LABEL_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    scene = np.load(v306.SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    train_data = np.load(v306.TRAIN_CANDIDATES)
    train_candidates = {key: train_data[key] for key in train_data.files}
    test_data = np.load(v306.TEST_CANDIDATES)
    test_candidates = {key: test_data[key] for key in test_data.files}
    train_raw, test_raw = v306.make_features(
        train_pos, test_pos, train_candidates, test_candidates, patches
    )
    train_h = candidate_h_center(train_pos, train_candidates)
    test_h = candidate_h_center(test_pos, test_candidates)
    labels, quality, label_report = v306.build_labels()
    targets = np.load(v306.TARGET_CACHE)
    truth_sh = targets["sh"].astype(np.float32)
    truth_d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_rows = []
    base_sh_parts = []
    model_sh_parts = []
    truth_sh_parts = []
    base_d_parts = []
    model_d_parts = []
    truth_d_parts = []
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
        inner_fit_raw = train_raw.take(inner_fit)
        inner_val_raw = train_raw.take(inner_ids)
        inner_fit_feature, _full, inner_val_feature, _scale = v264.standardize_features(
            inner_fit_raw, train_raw, inner_val_raw
        )
        inner_anchor_sh = generic_knn_anchor(inner_fit, inner_fit, train_pos, truth_sh)
        inner_anchor_d = generic_knn_anchor(inner_fit, inner_fit, train_pos, truth_d)
        inner_val_anchor_sh = generic_knn_anchor(inner_ids, inner_fit, train_pos, truth_sh)
        inner_val_anchor_d = generic_knn_anchor(inner_ids, inner_fit, train_pos, truth_d)
        _model, inner_report = fit_model(
            inner_fit_feature,
            train_h[inner_fit],
            inner_anchor_sh,
            inner_anchor_d,
            truth_sh[inner_fit],
            truth_d[inner_fit],
            labels[inner_fit],
            quality[inner_fit],
            seed,
            EPOCHS,
            len(patches),
            device,
            (
                inner_val_feature,
                train_h[inner_ids],
                inner_val_anchor_sh,
                inner_val_anchor_d,
                truth_sh[inner_ids],
                truth_d[inner_ids],
            ),
        )
        selected_epoch = int(inner_report["best_epoch"])
        selected_epochs.append(selected_epoch)

        outer_fit_raw = train_raw.take(outer_fit)
        outer_val_raw = train_raw.take(outer_ids)
        outer_fit_feature, _full, outer_val_feature, _scale = v264.standardize_features(
            outer_fit_raw, train_raw, outer_val_raw
        )
        outer_anchor_sh = generic_knn_anchor(outer_fit, outer_fit, train_pos, truth_sh)
        outer_anchor_d = generic_knn_anchor(outer_fit, outer_fit, train_pos, truth_d)
        outer_val_anchor_sh = generic_knn_anchor(outer_ids, outer_fit, train_pos, truth_sh)
        outer_val_anchor_d = generic_knn_anchor(outer_ids, outer_fit, train_pos, truth_d)
        if selected_epoch == 0:
            pred_sh = outer_val_anchor_sh.copy()
            pred_d = outer_val_anchor_d.copy()
            gate = np.zeros(len(outer_ids), dtype=np.float32)
            outer_training = {"best_epoch": 0}
        else:
            model, outer_training = fit_model(
                outer_fit_feature,
                train_h[outer_fit],
                outer_anchor_sh,
                outer_anchor_d,
                truth_sh[outer_fit],
                truth_d[outer_fit],
                labels[outer_fit],
                quality[outer_fit],
                seed + 50_000,
                selected_epoch,
                len(patches),
                device,
            )
            pred_sh, pred_d, gate = predict_joint(
                model,
                outer_val_feature,
                train_h[outer_ids],
                outer_val_anchor_d,
                outer_val_anchor_sh,
                device,
            )
        base_sh_score = float(np.mean(v297.row_cos(outer_val_anchor_sh, truth_sh[outer_ids])))
        base_d_score = float(np.mean(v297.row_cos(outer_val_anchor_d, truth_d[outer_ids])))
        model_sh_score = float(np.mean(v297.row_cos(pred_sh, truth_sh[outer_ids])))
        model_d_score = float(np.mean(v297.row_cos(pred_d, truth_d[outer_ids])))
        base_sh_parts.append(outer_val_anchor_sh)
        model_sh_parts.append(pred_sh)
        truth_sh_parts.append(truth_sh[outer_ids])
        base_d_parts.append(outer_val_anchor_d)
        model_d_parts.append(pred_d)
        truth_d_parts.append(truth_d[outer_ids])
        fold_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "selected_epoch": selected_epoch,
                "base_sh": base_sh_score,
                "base_d": base_d_score,
                "model_sh": model_sh_score,
                "model_d": model_d_score,
                "raw_weighted_delta": 0.52 * (model_sh_score - base_sh_score) + 0.48 * (model_d_score - base_d_score),
                "gate": summarize(gate),
                "inner": inner_report,
                "outer_training": outer_training,
            }
        )
        print(
            f"v308 fold {fold + 1}/5 epoch={selected_epoch} "
            f"d={model_d_score-base_d_score:+.6f} sh={model_sh_score-base_sh_score:+.6f} "
            f"weighted={fold_rows[-1]['raw_weighted_delta']:+.6f}", flush=True
        )

    base_sh_all = np.concatenate(base_sh_parts)
    model_sh_all = np.concatenate(model_sh_parts)
    truth_sh_all = np.concatenate(truth_sh_parts)
    base_d_all = np.concatenate(base_d_parts)
    model_d_all = np.concatenate(model_d_parts)
    truth_d_all = np.concatenate(truth_d_parts)
    selected, alpha_rows = choose_alphas(
        base_sh_all, model_sh_all, truth_sh_all,
        base_d_all, model_d_all, truth_d_all,
    )
    blended_sh = v297.normalize_rows(
        base_sh_all + selected["alpha_sh"] * (model_sh_all - base_sh_all)
    )
    blended_d = v297.normalize_rows(
        base_d_all + selected["alpha_d"] * (model_d_all - base_d_all)
    )
    base_sh_score = float(np.mean(v297.row_cos(base_sh_all, truth_sh_all)))
    base_d_score = float(np.mean(v297.row_cos(base_d_all, truth_d_all)))
    blended_sh_score = float(np.mean(v297.row_cos(blended_sh, truth_sh_all)))
    blended_d_score = float(np.mean(v297.row_cos(blended_d, truth_d_all)))
    base_weighted = 0.52 * base_sh_score + 0.48 * base_d_score
    blended_weighted = 0.52 * blended_sh_score + 0.48 * blended_d_score
    fold_deltas = []
    start = 0
    for row, base_sh, model_sh, truth_sh_fold, base_d, model_d, truth_d_fold in zip(
        fold_rows, base_sh_parts, model_sh_parts, truth_sh_parts,
        base_d_parts, model_d_parts, truth_d_parts,
    ):
        end = start + len(base_sh)
        current_sh = blended_sh[start:end]
        current_d = blended_d[start:end]
        sh_score = float(np.mean(v297.row_cos(current_sh, truth_sh_fold)))
        d_score = float(np.mean(v297.row_cos(current_d, truth_d_fold)))
        weighted = 0.52 * sh_score + 0.48 * d_score
        base = 0.52 * float(row["base_sh"]) + 0.48 * float(row["base_d"])
        row["blended_sh"] = sh_score
        row["blended_d"] = d_score
        row["blended_weighted_delta"] = weighted - base
        fold_deltas.append(weighted - base)
        start = end
    official_delta = blended_weighted - base_weighted
    official_selection = bool(
        official_delta >= MIN_OFFICIAL_JOINT_DELTA
        and sum(value > 0.0 for value in fold_deltas) >= 4
    )

    deployment = None
    if official_selection:
        full_raw = train_raw.take(valid_ids)
        full_feature, _unused, test_feature, _scale = v264.standardize_features(
            full_raw, train_raw, test_raw
        )
        train_anchor_sh = generic_knn_anchor(valid_ids, valid_ids, train_pos, truth_sh)
        train_anchor_d = generic_knn_anchor(valid_ids, valid_ids, train_pos, truth_d)
        tree = cKDTree(train_pos[valid_ids, :2])
        distance, local = tree.query(test_pos[:, :2], k=K_ANCHOR)
        weight = 1.0 / np.maximum(distance, 0.05) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        test_anchor_sh = v297.normalize_rows(
            np.sum(weight[:, :, None, None] * truth_sh[valid_ids[local]], axis=1, dtype=np.float64)
        )
        test_anchor_d = v297.normalize_rows(
            np.sum(weight[:, :, None] * truth_d[valid_ids[local]], axis=1, dtype=np.float64)
        )
        final_epochs = max(1, int(np.median([value for value in selected_epochs if value > 0])))
        sh_predictions = []
        d_predictions = []
        checkpoints = []
        for seed in FINAL_SEEDS:
            model, _training = fit_model(
                full_feature,
                train_h[valid_ids],
                train_anchor_sh,
                train_anchor_d,
                truth_sh[valid_ids],
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
            pred_sh, pred_d, _gate = predict_joint(
                model, test_feature, test_h, test_anchor_d, test_anchor_sh, device
            )
            sh_predictions.append(pred_sh)
            d_predictions.append(pred_d)
        model_test_sh = v297.normalize_rows(np.mean(sh_predictions, axis=0))
        model_test_d = v297.normalize_rows(np.mean(d_predictions, axis=0))
        target_sh = v297.normalize_rows(
            test_anchor_sh + selected["alpha_sh"] * (model_test_sh - test_anchor_sh)
        )
        target_d = v297.normalize_rows(
            test_anchor_d + selected["alpha_d"] * (model_test_d - test_anchor_d)
        )
        np.savez_compressed(TARGETS_OUT, sh=target_sh, d=target_d)
        diagnostics = v266.alternating_project(v306.BASE, OUTPUT, target_sh, target_d)
        deployment = {
            "final_epochs": final_epochs,
            "checkpoints": checkpoints,
            "targets": PATHS.relative(TARGETS_OUT),
            "prediction": PATHS.relative(OUTPUT),
            "local_score": v208.score_prediction(OUTPUT),
            "diagnostics": {key: summarize(value) for key, value in diagnostics.items()},
        }

    report = {
        "method": VERSION,
        "status": "official_selection_passed" if official_selection else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The v305 identity-matched path energy should jointly improve absolute-delay PDP "
            "and BS departure S-H when both are rendered from the same physical path set."
        ),
        "label_report": label_report,
        "fold_rows": fold_rows,
        "oof": {
            "base_sh": base_sh_score,
            "base_d": base_d_score,
            "base_weighted": base_weighted,
            "selected": selected,
            "blended_sh": blended_sh_score,
            "blended_d": blended_d_score,
            "blended_weighted": blended_weighted,
            "delta": official_delta,
            "fold_deltas": fold_deltas,
            "fold_wins": int(sum(value > 0.0 for value in fold_deltas)),
            "minimum_required_delta": MIN_OFFICIAL_JOINT_DELTA,
            "official_selection": official_selection,
            "alpha_grid": alpha_rows,
        },
        "deployment": deployment,
        "outputs": {
            "report": PATHS.relative(REPORT),
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["oof"], indent=2), flush=True)


if __name__ == "__main__":
    main()
