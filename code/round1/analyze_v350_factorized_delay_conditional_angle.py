#!/usr/bin/env python3
"""Factorize global delay mass from conditional H-angle allocation.

This is an official-only analysis.  The model receives a strict cross-fit
v239-like current and is only allowed to redistribute H-angle energy inside
each fixed S/delay row.  It cannot change the current PDP marginal.
"""

from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn.functional as F

import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v279_joint_component_trajectory as v279
import analyze_v283_joint_geometry_path_renderer as v283
import analyze_v284_planar_geometry_residual_field as v284
import analyze_v285_v239_isomorphic_joint_residual as v285
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 350
VERSION = "v350_factorized_delay_conditional_angle"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v350_factorized_oof.npz"
SAMPLES = PATHS.cache_dir(285) / "v285_v239_crossfit_joint_current.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

FOLD_COUNT = 5
MAX_EPOCHS = 30
EVAL_EVERY = 5
BATCH_SIZE = 24
ALPHA_GRID = np.linspace(0.0, 1.25, 11, dtype=np.float32)
MIN_MEAN_DELTA = 0.01
MIN_FOLD_WINS = 4
FLOOR = 1e-10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def impose_s_marginal_torch(
    prediction_flat: torch.Tensor, anchor_flat: torch.Tensor
) -> torch.Tensor:
    prediction = prediction_flat.reshape(-1, v279.S_SIZE, v279.H_SIZE)
    anchor = anchor_flat.reshape(-1, v279.S_SIZE, v279.H_SIZE)
    conditional = prediction / prediction.sum(dim=2, keepdim=True).clamp_min(FLOOR)
    anchor_s = anchor.sum(dim=2, keepdim=True)
    output = conditional * anchor_s
    return output / output.flatten(1).sum(dim=1)[:, None, None].clamp_min(1e-30)


def impose_s_marginal_numpy(prediction: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    pred = np.maximum(np.asarray(prediction, dtype=np.float32), 0.0)
    base = np.maximum(np.asarray(anchor, dtype=np.float32), 0.0)
    conditional = pred / np.maximum(pred.sum(axis=2, keepdims=True), FLOOR)
    output = conditional * base.sum(axis=2, keepdims=True)
    return v279.normalize(output)


def conditional_bc(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = np.maximum(np.asarray(prediction, dtype=np.float32), 0.0)
    target = np.maximum(np.asarray(truth, dtype=np.float32), 0.0)
    pred_s = pred.sum(axis=2)
    truth_s = target.sum(axis=2)
    pred_q = pred / np.maximum(pred_s[:, :, None], FLOOR)
    truth_q = target / np.maximum(truth_s[:, :, None], FLOOR)
    overlap = np.sqrt(np.maximum(pred_s * truth_s, 0.0))
    overlap /= np.maximum(overlap.sum(axis=1, keepdims=True), FLOOR)
    bc = np.sum(np.sqrt(np.maximum(pred_q * truth_q, 0.0)), axis=2)
    return np.sum(overlap * bc, axis=1).astype(np.float32)


def conditional_loss(
    prediction_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    truth_flat: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> torch.Tensor:
    prediction = impose_s_marginal_torch(prediction_flat, anchor_flat)
    truth = truth_flat.reshape(-1, v279.S_SIZE, v279.H_SIZE)
    anchor = anchor_flat.reshape(-1, v279.S_SIZE, v279.H_SIZE)

    pred_s = prediction.sum(dim=2).clamp_min(FLOOR)
    truth_s = truth.sum(dim=2).clamp_min(FLOOR)
    pred_q = prediction / pred_s[:, :, None]
    truth_q = truth / truth_s[:, :, None]
    overlap = torch.sqrt(pred_s * truth_s)
    overlap = overlap / overlap.sum(dim=1, keepdim=True).clamp_min(FLOOR)
    bc = torch.sum(
        torch.sqrt(pred_q.clamp_min(FLOOR) * truth_q.clamp_min(FLOOR)), dim=2
    )
    bc_loss = 1.0 - torch.sum(overlap * bc, dim=1).mean()

    target_fixed = truth_q * anchor.sum(dim=2, keepdim=True)
    target_fixed = target_fixed / target_fixed.flatten(1).sum(dim=1)[:, None, None]
    joint_loss = 1.0 - F.cosine_similarity(
        prediction.flatten(1), target_fixed.flatten(1), dim=1
    ).mean()
    h_loss = 1.0 - F.cosine_similarity(
        prediction.sum(dim=1), target_fixed.sum(dim=1), dim=1
    ).mean()
    regularizer = (
        2e-6 * diagnostics["correction"].square().mean()
        + 2e-5 * diagnostics["source_mass"].mean()
    )
    return 0.52 * bc_loss + 0.38 * joint_loss + 0.10 * h_loss + regularizer


@torch.no_grad()
def predict(
    model: v284.PlanarGeometryResidualField,
    feature: v264.FeatureSet,
    center_s: np.ndarray,
    center_h: np.ndarray,
    anchor: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty_like(anchor, dtype=np.float32)
    gates = np.empty(len(anchor), dtype=np.float32)
    for start in range(0, len(anchor), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(anchor))
        anchor_tensor = torch.from_numpy(anchor[start:end]).to(device)
        raw, diagnostics = model(
            torch.from_numpy(feature.global_features[start:end]).to(device),
            torch.from_numpy(feature.path_numeric[start:end]).to(device),
            torch.from_numpy(feature.patch_id[start:end]).to(device),
            torch.from_numpy(feature.path_type[start:end]).to(device),
            torch.from_numpy(feature.path_valid[start:end]).to(device),
            torch.from_numpy(center_s[start:end]).to(device),
            torch.from_numpy(center_h[start:end]).to(device),
            anchor_tensor,
        )
        output[start:end] = impose_s_marginal_torch(raw, anchor_tensor).cpu().numpy()
        gates[start:end] = diagnostics["source_mass"].cpu().numpy()
    return v279.normalize(output), gates


def choose_alpha(
    base: np.ndarray, model: np.ndarray, truth: np.ndarray, truth_d: np.ndarray
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for alpha in ALPHA_GRID:
        value = v279.normalize(base + float(alpha) * (model - base))
        rows.append(
            {
                "alpha": float(alpha),
                **v279.metrics(value, truth, truth_d),
                "conditional_bc": float(np.mean(conditional_bc(value, truth))),
            }
        )
    best = max(rows, key=lambda row: (row["weighted"], row["conditional_bc"]))
    return float(best["alpha"]), best, rows


def train_model(
    feature: v264.FeatureSet,
    center_s: np.ndarray,
    center_h: np.ndarray,
    anchor: np.ndarray,
    truth: np.ndarray,
    truth_d: np.ndarray,
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
) -> tuple[v284.PlanarGeometryResidualField, dict[str, object]]:
    set_seed(seed)
    model = v284.PlanarGeometryResidualField(
        feature.global_features.shape[1], feature.path_numeric.shape[2], patch_count
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.8e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = 0
    best_alpha = 0.0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    history: list[dict[str, object]] = []
    if validation is not None:
        _, _, _, val_anchor, val_truth, val_truth_d = validation
        initial = val_anchor.copy()
        _, initial_best, _ = choose_alpha(
            val_anchor, initial, val_truth, val_truth_d
        )
        best_metric = float(initial_best["weighted"])

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(anchor))
        losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            anchor_tensor = torch.from_numpy(anchor[ids]).to(device)
            raw, diagnostics = model(
                torch.from_numpy(feature.global_features[ids]).to(device),
                torch.from_numpy(feature.path_numeric[ids]).to(device),
                torch.from_numpy(feature.patch_id[ids]).to(device),
                torch.from_numpy(feature.path_type[ids]).to(device),
                torch.from_numpy(feature.path_valid[ids]).to(device),
                torch.from_numpy(center_s[ids]).to(device),
                torch.from_numpy(center_h[ids]).to(device),
                anchor_tensor,
            )
            loss = conditional_loss(
                raw,
                anchor_tensor,
                torch.from_numpy(truth[ids].reshape(len(ids), -1)).to(device),
                diagnostics,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs:
            item: dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
            }
            if validation is not None:
                val_feature, val_s, val_h, val_anchor, val_truth, val_truth_d = validation
                val_model, val_gates = predict(
                    model, val_feature, val_s, val_h, val_anchor, device
                )
                alpha, selected, _ = choose_alpha(
                    val_anchor, val_model, val_truth, val_truth_d
                )
                item.update(
                    {
                        "validation": selected,
                        "selected_alpha": alpha,
                        "gate": summarize(val_gates),
                    }
                )
                if selected["weighted"] > best_metric:
                    best_metric = float(selected["weighted"])
                    best_epoch = epoch
                    best_alpha = alpha
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"v350 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_alpha": float(best_alpha),
        "best_weighted": float(best_metric),
        "history": history,
    }


def main() -> None:
    for path in (
        SAMPLES,
        TARGET_CACHE,
        v283.TRAIN_CANDIDATES,
        v283.TEST_CANDIDATES,
        v283.SCENE_CACHE,
        v283.V265_REPORT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    sample = np.load(SAMPLES)
    query_ids = sample["query_ids"].astype(np.int64)
    assigned_fold = sample["assigned_fold"].astype(np.int16)
    joint_anchor = sample["reconstructed_anchor_sh"].astype(np.float32)
    delay_current = sample["current_sh"].astype(np.float32)
    target_data = np.load(TARGET_CACHE)
    truth_sh = v279.normalize(target_data["sh"].astype(np.float32))
    truth_d = v279.normalize(target_data["d"].astype(np.float32))

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    patches = np.load(v283.SCENE_CACHE)["wall_patches"].astype(np.float32)
    train_candidates = dict(np.load(v283.TRAIN_CANDIDATES))
    test_candidates = dict(np.load(v283.TEST_CANDIDATES))
    train_node, test_node = v257.build_node_features(train_pos[:, :2], test_pos[:, :2])
    train_raw = v264.build_features(train_pos, train_candidates, patches, train_node)
    test_raw = v264.build_features(test_pos, test_candidates, patches, test_node)
    with v283.V265_REPORT.open("r", encoding="utf-8") as handle:
        center_spec = json.load(handle)["selection"]["best"]
    train_s, train_h = v283.centers(train_pos, train_candidates, train_raw, center_spec)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v350 device={device} samples={len(query_ids)}", flush=True)

    fold_rows: list[dict[str, object]] = []
    oof_ids: list[np.ndarray] = []
    oof_base: list[np.ndarray] = []
    oof_model: list[np.ndarray] = []
    for fold in range(FOLD_COUNT):
        outer_rows = np.flatnonzero(assigned_fold == fold)
        inner_fold = (fold + 1) % FOLD_COUNT
        inner_rows = np.flatnonzero(assigned_fold == inner_fold)
        fit_rows = np.flatnonzero(
            (assigned_fold != fold) & (assigned_fold != inner_fold)
        )

        fit_ids = query_ids[fit_rows]
        fit_feature, full_feature, _test_feature, _ = v264.standardize_features(
            train_raw.take(fit_ids), train_raw, test_raw
        )
        inner_feature = full_feature.take(query_ids[inner_rows])
        _, inner_training = train_model(
            fit_feature,
            train_s[fit_ids],
            train_h[fit_ids],
            delay_current[fit_rows],
            truth_sh[fit_ids],
            truth_d[fit_ids],
            350 + fold,
            MAX_EPOCHS,
            device,
            len(patches),
            (
                inner_feature,
                train_s[query_ids[inner_rows]],
                train_h[query_ids[inner_rows]],
                delay_current[inner_rows],
                truth_sh[query_ids[inner_rows]],
                truth_d[query_ids[inner_rows]],
            ),
        )
        epoch = int(inner_training["best_epoch"])
        alpha = float(inner_training["best_alpha"])

        train_rows = np.flatnonzero(assigned_fold != fold)
        train_ids = query_ids[train_rows]
        outer_train_feature, outer_full_feature, _test_feature, _ = (
            v264.standardize_features(train_raw.take(train_ids), train_raw, test_raw)
        )
        outer_feature = outer_full_feature.take(query_ids[outer_rows])
        if epoch == 0 or alpha == 0.0:
            raw_model = delay_current[outer_rows].copy()
            prediction = raw_model.copy()
            gates = np.zeros(len(outer_rows), dtype=np.float32)
            outer_training: dict[str, object] = {
                "seed": 103_500 + fold,
                "best_epoch": 0,
                "history": [],
            }
        else:
            model, outer_training = train_model(
                outer_train_feature,
                train_s[train_ids],
                train_h[train_ids],
                delay_current[train_rows],
                truth_sh[train_ids],
                truth_d[train_ids],
                103_500 + fold,
                epoch,
                device,
                len(patches),
                None,
            )
            raw_model, gates = predict(
                model,
                outer_feature,
                train_s[query_ids[outer_rows]],
                train_h[query_ids[outer_rows]],
                delay_current[outer_rows],
                device,
            )
            prediction = v279.normalize(
                delay_current[outer_rows]
                + alpha * (raw_model - delay_current[outer_rows])
            )

        ids = query_ids[outer_rows]
        joint_metric = v279.metrics(joint_anchor[outer_rows], truth_sh[ids], truth_d[ids])
        delay_metric = v279.metrics(delay_current[outer_rows], truth_sh[ids], truth_d[ids])
        model_metric = v279.metrics(prediction, truth_sh[ids], truth_d[ids])
        delta = {
            key: float(model_metric[key] - delay_metric[key])
            for key in ("sh", "d", "weighted")
        }
        d_identity = np.sum(
            np.abs(
                v279.derive_d_numpy(prediction)
                - v279.derive_d_numpy(delay_current[outer_rows])
            ),
            axis=1,
        )
        fold_rows.append(
            {
                "fold": fold,
                "inner_fold": inner_fold,
                "fit_count": int(len(fit_rows)),
                "inner_count": int(len(inner_rows)),
                "outer_count": int(len(outer_rows)),
                "selected_epoch": epoch,
                "selected_alpha": alpha,
                "inner_training": inner_training,
                "outer_training": outer_training,
                "joint_knn_anchor": joint_metric,
                "delay_only": delay_metric,
                "factorized": model_metric,
                "factorized_delta_vs_delay_only": delta,
                "conditional_bc": {
                    "joint_knn_anchor": float(
                        np.mean(conditional_bc(joint_anchor[outer_rows], truth_sh[ids]))
                    ),
                    "delay_only": float(
                        np.mean(conditional_bc(delay_current[outer_rows], truth_sh[ids]))
                    ),
                    "factorized": float(np.mean(conditional_bc(prediction, truth_sh[ids]))),
                },
                "pdp_identity_l1": summarize(d_identity),
                "gate": summarize(gates),
            }
        )
        oof_ids.append(ids)
        oof_base.append(delay_current[outer_rows])
        oof_model.append(prediction)
        print(
            f"v350 fold={fold} epoch={epoch} alpha={alpha:.3f} "
            f"delta={delta['weighted']:+.6f}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_delta = {
        key: float(
            np.mean(
                [row["factorized_delta_vs_delay_only"][key] for row in fold_rows]
            )
        )
        for key in ("sh", "d", "weighted")
    }
    wins = int(
        sum(row["factorized_delta_vs_delay_only"]["weighted"] > 0.0 for row in fold_rows)
    )
    decision = bool(mean_delta["weighted"] >= MIN_MEAN_DELTA and wins >= MIN_FOLD_WINS)
    ids_all = np.concatenate(oof_ids)
    base_all = np.concatenate(oof_base)
    model_all = np.concatenate(oof_model)
    np.savez_compressed(
        OOF_OUT,
        ids=ids_all,
        delay_only=base_all,
        factorized=model_all,
    )
    report = {
        "method": VERSION,
        "status": "invalidated_crossfold_current_leakage",
        "round": 3,
        "validity_audit": {
            "strict_oof_usable": False,
            "invalidated_by": "v353_crossfold_current_leakage_audit",
            "reason": "reused v285 currents contain labels from the new outer fold",
            "numerical_decision_before_audit": decision,
        },
        "primary_hypothesis": (
            "The validated global-delay current and conditional H-angle allocation are separate "
            "prediction problems; learning P(H|D) while exactly freezing P(D) should recover "
            "joint support without the destructive alternating projections seen in v338/v339."
        ),
        "compliance": {
            "uses": [
                "official train H joint S-H/PDP labels",
                "official train coordinates",
                "official Round1_Map point-cloud wall-path features",
                "strict fit-only v239-like cross-fit delay current",
            ],
            "does_not_use": [
                "v207/v212/synthetic test truth",
                "restored Etoile scene, material, or RT coefficients",
                "online labels or local replay truth",
            ],
        },
        "parameters": {
            "fold_count": FOLD_COUNT,
            "maximum_inner_epochs": MAX_EPOCHS,
            "alpha_grid": ALPHA_GRID.tolist(),
            "minimum_mean_delta": MIN_MEAN_DELTA,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "selection": "epoch and alpha selected only on the disjoint inner fold",
            "fixed_quantity": "per-point S/global-delay marginal",
        },
        "fold_rows": fold_rows,
        "decision": {
            "mean_delta": mean_delta,
            "fold_wins": wins,
            "passed": False,
            "passed_numerically_before_audit": decision,
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
