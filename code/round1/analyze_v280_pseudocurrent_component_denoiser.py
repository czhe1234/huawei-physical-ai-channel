#!/usr/bin/env python3
"""Official-supervised component denoising in the current/test spectrum domain."""

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
import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 280
VERSION = "v280_pseudocurrent_component_denoiser"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v280_pseudocurrent_targets.npz"
OUTPUT = OUT_DIR / "v280_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

LOCAL_GATE = 0.645604
OUTER_SEEDS = (280, 1280, 2280, 3280, 4280)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 50
EVAL_EVERY = 5
BATCH_SIZE = 64
MIN_OFFICIAL_DELTA = 5e-4
FINAL_SEEDS = (28001, 28031, 28061)
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pseudo_from_fit(raw: v274.RawSet, truth_sh: np.ndarray) -> np.ndarray:
    prediction = np.sum(
        raw.base_weights[:, :, None, None] * truth_sh[raw.neighbor_ids],
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    return v279.normalize(prediction)


@torch.no_grad()
def predict_model(
    model: v279.ComponentTrajectoryNet,
    data: v274.ModelSet,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
    current_sh: np.ndarray,
    peak_indices: np.ndarray,
    peak_mass: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    node = v279.standardized_nodes(node_raw, stats)
    output = np.empty(
        (len(data.query_ids), v279.S_SIZE, v279.H_SIZE), dtype=np.float32
    )
    gates = np.empty(len(data.query_ids), dtype=np.float32)
    for start in range(0, len(data.query_ids), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(data.query_ids))
        query_ids = data.query_ids[start:end]
        neighbor_ids = data.neighbor_ids[start:end]
        prediction, diagnostics = model(
            torch.from_numpy(data.query_node[start:end]).to(device),
            torch.from_numpy(node[neighbor_ids]).to(device),
            torch.from_numpy(data.edge[start:end]).to(device),
            torch.from_numpy(data.base_weights[start:end]).to(device),
            torch.from_numpy(current_sh[neighbor_ids]).to(device),
            torch.from_numpy(peak_indices[neighbor_ids]).to(device),
            torch.from_numpy(peak_mass[neighbor_ids]).to(device),
            query_anchor=torch.from_numpy(current_sh[query_ids]).to(device),
        )
        output[start:end] = prediction.reshape(
            -1, v279.S_SIZE, v279.H_SIZE
        ).cpu().numpy()
        gates[start:end] = diagnostics["gate"].cpu().numpy()
    return v279.normalize(output), gates


def train_model(
    train_data: v274.ModelSet,
    validation: v274.ModelSet | None,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
    current_sh: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
    peak_indices: np.ndarray,
    peak_mass: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[v279.ComponentTrajectoryNet, dict[str, object]]:
    set_seed(seed)
    model = v279.ComponentTrajectoryNet(
        node_raw.shape[1], train_data.edge.shape[2]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    node = v279.standardized_nodes(node_raw, stats)
    rng = np.random.default_rng(seed)
    baseline = None
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    if validation is not None:
        baseline = v279.metrics(
            current_sh[validation.query_ids],
            truth_sh[validation.query_ids],
            truth_d[validation.query_ids],
        )
        best_metric = baseline["weighted"]
        best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_data.query_ids))
        losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            query_ids = train_data.query_ids[rows]
            neighbor_ids = train_data.neighbor_ids[rows]
            prediction, diagnostics = model(
                torch.from_numpy(train_data.query_node[rows]).to(device),
                torch.from_numpy(node[neighbor_ids]).to(device),
                torch.from_numpy(train_data.edge[rows]).to(device),
                torch.from_numpy(train_data.base_weights[rows]).to(device),
                torch.from_numpy(current_sh[neighbor_ids]).to(device),
                torch.from_numpy(peak_indices[neighbor_ids]).to(device),
                torch.from_numpy(peak_mass[neighbor_ids]).to(device),
                query_anchor=torch.from_numpy(current_sh[query_ids]).to(device),
            )
            loss = v279.model_loss(
                prediction,
                torch.from_numpy(
                    truth_sh[query_ids].reshape(len(rows), -1)
                ).to(device),
                torch.from_numpy(truth_d[query_ids]).to(device),
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
                prediction, gates = predict_model(
                    model,
                    validation,
                    node_raw,
                    stats,
                    current_sh,
                    peak_indices,
                    peak_mass,
                    device,
                )
                validation_metric = v279.metrics(
                    prediction,
                    truth_sh[validation.query_ids],
                    truth_d[validation.query_ids],
                )
                item["validation"] = validation_metric
                item["gate"] = summarize(gates)
                item["baseline"] = baseline
                if validation_metric["weighted"] > best_metric:
                    best_metric = validation_metric["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"v280 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
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


def make_current(
    node_count: int,
    assignments: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    current = np.zeros(
        (node_count, v279.S_SIZE, v279.H_SIZE), dtype=np.float32
    )
    for ids, values in assignments:
        current[ids] = values
    return current


def model_sets(
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    direct_bank_ids: np.ndarray,
    all_pos: np.ndarray,
    signatures: np.ndarray,
    blocker_count: np.ndarray,
    footprint: np.ndarray,
    z_max: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    node_raw: np.ndarray,
    truth_sh: np.ndarray,
) -> tuple[
    v274.ModelSet,
    v274.ModelSet,
    v274.Standardizer,
    np.ndarray,
]:
    raw_train = v279.build_sets(
        train_ids,
        train_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )
    raw_direct_validation = v279.build_sets(
        validation_ids,
        direct_bank_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )
    raw_validation_cohort = v279.build_sets(
        validation_ids,
        validation_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )
    pseudo_train = pseudo_from_fit(raw_train, truth_sh)
    pseudo_validation = pseudo_from_fit(raw_direct_validation, truth_sh)
    current = make_current(
        len(all_pos),
        ((train_ids, pseudo_train), (validation_ids, pseudo_validation)),
    )
    stats = v274.fit_standardizer(node_raw, train_ids, raw_train)
    return (
        v274.apply_standardizer(raw_train, node_raw, stats),
        v274.apply_standardizer(raw_validation_cohort, node_raw, stats),
        stats,
        current,
    )


def main() -> None:
    for path in (BASE, TARGET_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_data = np.load(TARGET_CACHE)
    truth_sh = v279.normalize(target_data["sh"].astype(np.float32))
    truth_d = v279.normalize(target_data["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
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
    train_pos = all_pos[: len(truth_sh)]
    test_pos = all_pos[len(truth_sh) :]
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v280 device={device} valid={len(valid_ids)}", flush=True)

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

        inner_train, inner_val, inner_stats, inner_current = model_sets(
            inner_fit,
            inner_ids,
            inner_fit,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
            node_raw,
            truth_sh,
        )
        inner_peaks, inner_mass = v279.top_peaks(inner_current)
        _, inner_training = train_model(
            inner_train,
            inner_val,
            node_raw,
            inner_stats,
            inner_current,
            truth_sh,
            truth_d,
            inner_peaks,
            inner_mass,
            seed,
            EPOCHS,
            device,
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)

        outer_train, outer_val, outer_stats, outer_current = model_sets(
            outer_fit,
            outer_ids,
            outer_fit,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
            node_raw,
            truth_sh,
        )
        outer_peaks, outer_mass = v279.top_peaks(outer_current)
        base_sh = outer_current[outer_ids]
        if selected_epoch == 0:
            model_sh = base_sh.copy()
            gates = np.zeros(len(outer_ids), dtype=np.float32)
            outer_training = {
                "seed": seed + 100_000,
                "best_epoch": 0,
                "history": [],
            }
        else:
            outer_model, outer_training = train_model(
                outer_train,
                None,
                node_raw,
                outer_stats,
                outer_current,
                truth_sh,
                truth_d,
                outer_peaks,
                outer_mass,
                seed + 100_000,
                selected_epoch,
                device,
            )
            model_sh, gates = predict_model(
                outer_model,
                outer_val,
                node_raw,
                outer_stats,
                outer_current,
                outer_peaks,
                outer_mass,
                device,
            )
        baseline_metric = v279.metrics(
            base_sh, truth_sh[outer_ids], truth_d[outer_ids]
        )
        model_metric = v279.metrics(
            model_sh, truth_sh[outer_ids], truth_d[outer_ids]
        )
        raw_delta = {
            key: model_metric[key] - baseline_metric[key]
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
                "baseline": baseline_metric,
                "model": model_metric,
                "raw_delta": raw_delta,
                "gate": summarize(gates),
            }
        )
        oof_ids_parts.append(outer_ids)
        oof_base_parts.append(base_sh)
        oof_model_parts.append(model_sh)
        print(
            f"v280 outer {fold + 1}/{len(OUTER_SEEDS)} epoch={selected_epoch} "
            f"base={baseline_metric['weighted']:.6f} "
            f"model={model_metric['weighted']:.6f} delta={raw_delta['weighted']:+.6f}",
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
        blended_metric = v279.metrics(
            blended, truth_sh[ids], truth_d[ids]
        )
        delta = {
            key: blended_metric[key] - row["baseline"][key]
            for key in ("sh", "d", "weighted")
        }
        row["blended"] = blended_metric
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
        test_ids = np.arange(len(truth_sh), len(all_pos), dtype=np.int64)
        raw_full_train = v279.build_sets(
            valid_ids,
            valid_ids,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        raw_test_cohort = v279.build_sets(
            test_ids,
            test_ids,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        pseudo_train = pseudo_from_fit(raw_full_train, truth_sh)
        current_test, _ = v274.extract_current_marginals(BASE)
        current = make_current(
            len(all_pos),
            ((valid_ids, pseudo_train), (test_ids, current_test)),
        )
        full_stats = v274.fit_standardizer(
            node_raw, valid_ids, raw_full_train
        )
        full_train = v274.apply_standardizer(
            raw_full_train, node_raw, full_stats
        )
        test_set = v274.apply_standardizer(
            raw_test_cohort, node_raw, full_stats
        )
        peaks, mass = v279.top_peaks(current)
        model_parts: list[np.ndarray] = []
        gate_parts: list[np.ndarray] = []
        for seed in FINAL_SEEDS:
            model, training = train_model(
                full_train,
                None,
                node_raw,
                full_stats,
                current,
                truth_sh,
                truth_d,
                peaks,
                mass,
                seed,
                final_epoch,
                device,
            )
            test_model, test_gate = predict_model(
                model,
                test_set,
                node_raw,
                full_stats,
                current,
                peaks,
                mass,
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
        target_sh = v279.normalize(
            current_test + alpha * (test_model - current_test)
        )
        target_d = v279.derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            current_sh=current_test,
            model_sh=test_model,
            target_sh=target_sh,
            target_d=target_d,
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
            "v279 learned the right discrete component mechanism but failed through cross-domain "
            "residual transfer; training the same mechanism as a pseudo-current denoiser and "
            "deploying it directly on the v239 test cohort should preserve the current domain."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H truth and fit-only LOO pseudo-current spectra",
                "official train/test coordinates",
                "official point-cloud wall/component/height features",
                "strict v239 official-only test current spectrum",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher outputs",
                "hidden labels",
                "RT MIMO H or path coefficients",
                "outer validation H in pseudo-current construction",
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
            "neighbors": v274.K_NEIGHBORS,
            "peaks_per_neighbor": v279.PEAK_COUNT,
            "offsets": [list(value) for value in v279.OFFSETS],
            "epochs": EPOCHS,
            "alpha_grid": ALPHA_GRID.tolist(),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
            "final_seeds": list(FINAL_SEEDS),
        },
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
