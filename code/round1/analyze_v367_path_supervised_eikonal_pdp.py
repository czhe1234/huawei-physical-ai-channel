#!/usr/bin/env python3
"""Path-supervised query-local eikonal PDP transport.

v366 showed that local eikonal surfaces retain nearly all true delay support but
an indirect spectrum loss does not learn surface existence or energy. This
experiment keeps the representation and model fixed while adding fit-only
component mass, delay residual, and action-gate supervision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v238_raw_path_conditional_angle as v238
import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v365_multiview_component_pdp as v365
import analyze_v366_local_eikonal_pdp as v366
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 367
VERSION = "v367_path_supervised_eikonal_pdp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_OUT = OUT_DIR / "v367_path_supervised_eikonal_pdp_oof.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

MAX_EPOCHS = 36
EVAL_EVERY = 2
TRAIN_BATCH = 24
LEARNING_RATE = 3.5e-4
MASS_LOSS_WEIGHT = 0.08
SHIFT_LOSS_WEIGHT = 0.03
GATE_LOSS_WEIGHT = 0.20
SHIFT_SIGMA = 0.18
GATE_GRID = np.linspace(0.0, 0.80, 17, dtype=np.float32)

MIN_COSINE_LIFT = 0.010
MIN_TOP_RECALL_LIFT = 0.05
MIN_EMD_REDUCTION = 0.10
MIN_FOLD_WINS = 4
MIN_PATH_SUPERVISION_LIFT = 0.002
MAX_CLEAR_DEGRADATION = 0.002


@dataclass
class PathData:
    query_ids: np.ndarray
    query_node: np.ndarray
    delay: np.ndarray
    feature: np.ndarray
    prior: np.ndarray
    valid: np.ndarray
    anchor: np.ndarray
    target_mass: np.ndarray
    target_shift: np.ndarray
    target_gate: np.ndarray
    covered_mass: np.ndarray


def build_supervision(
    hypothesis: v366.HypothesisSet,
    anchor: np.ndarray,
    truth_d: np.ndarray,
    component: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(hypothesis.query_ids)
    target_mass = np.zeros((count, v366.HYPOTHESES), dtype=np.float32)
    target_shift = np.zeros(
        (count, v366.HYPOTHESES, len(v366.OFFSETS)), dtype=np.float32
    )
    target_gate = np.zeros(count, dtype=np.float32)
    covered_mass = np.zeros(count, dtype=np.float32)
    offsets = np.asarray(v366.OFFSETS, dtype=np.float64)
    center = int(np.argmin(np.abs(offsets)))

    for row, point_id in enumerate(hypothesis.query_ids):
        keep = hypothesis.valid[row]
        surface_ids = np.flatnonzero(keep)
        if len(surface_ids) == 0:
            continue
        surface_delay = hypothesis.delay[row, surface_ids]
        active = np.flatnonzero(component["valid"][point_id])
        total_component_mass = float(
            np.sum(component["weight"][point_id, active], dtype=np.float64)
        )
        shift_accumulator = np.zeros(
            (len(surface_ids), len(v366.OFFSETS)), dtype=np.float64
        )
        local_mass = np.zeros(len(surface_ids), dtype=np.float64)
        for component_id in active:
            weight = float(component["weight"][point_id, component_id])
            distance = v365.circular_distance(
                surface_delay, component["delay"][point_id, component_id]
            )
            surface = int(np.argmin(distance))
            if float(distance[surface]) > 0.25:
                continue
            residual = float(
                v366.signed_circular_delta(
                    component["delay"][point_id, component_id],
                    surface_delay[surface],
                )
            )
            shift_probability = np.exp(
                -0.5 * ((offsets - residual) / SHIFT_SIGMA) ** 2
            )
            shift_probability /= max(float(np.sum(shift_probability)), 1e-20)
            local_mass[surface] += weight
            shift_accumulator[surface] += weight * shift_probability
        covered = float(np.sum(local_mass))
        covered_mass[row] = covered / max(total_component_mass, 1e-20)
        if covered <= 1e-20:
            local_mass = hypothesis.prior[row, surface_ids].astype(np.float64)
            shift_accumulator[:, center] = local_mass
        local_mass /= max(float(np.sum(local_mass)), 1e-20)
        target_mass[row, surface_ids] = local_mass.astype(np.float32)
        for local, surface_id in enumerate(surface_ids):
            total = float(np.sum(shift_accumulator[local]))
            if total > 1e-20:
                target_shift[row, surface_id] = (
                    shift_accumulator[local] / total
                ).astype(np.float32)
            else:
                target_shift[row, surface_id, center] = 1.0

        oracle = v297.dirichlet_atoms(surface_delay) @ local_mass
        oracle = v365.normalize_rows(oracle[None, :])[0]
        scores = []
        for gate in GATE_GRID:
            prediction = v365.normalize_rows(
                anchor[row : row + 1]
                + float(gate) * (oracle[None, :] - anchor[row : row + 1])
            )
            scores.append(
                float(v365.row_cos(prediction, truth_d[point_id : point_id + 1])[0])
            )
        target_gate[row] = GATE_GRID[int(np.argmax(scores))]
    return target_mass, target_shift, target_gate, covered_mass


def make_path_data(
    hypothesis: v366.HypothesisSet,
    query_node: np.ndarray,
    truth_d: np.ndarray,
    component: dict[str, np.ndarray],
    with_supervision: bool,
) -> PathData:
    base = v366.model_data(hypothesis, query_node, truth_d)
    if with_supervision:
        target_mass, target_shift, target_gate, covered_mass = build_supervision(
            hypothesis, base.anchor, truth_d, component
        )
    else:
        target_mass = np.zeros_like(base.prior)
        target_shift = np.zeros(
            base.prior.shape + (len(v366.OFFSETS),), dtype=np.float32
        )
        target_gate = np.zeros(len(base.query_ids), dtype=np.float32)
        covered_mass = np.zeros(len(base.query_ids), dtype=np.float32)
    return PathData(
        query_ids=base.query_ids,
        query_node=base.query_node,
        delay=base.delay,
        feature=base.feature,
        prior=base.prior,
        valid=base.valid,
        anchor=base.anchor,
        target_mass=target_mass,
        target_shift=target_shift,
        target_gate=target_gate,
        covered_mass=covered_mass,
    )


def shuffled_supervision(data: PathData, seed: int) -> PathData:
    rng = np.random.default_rng(seed)
    target_mass = np.zeros_like(data.target_mass)
    target_shift = np.zeros_like(data.target_shift)
    for row in range(len(data.query_ids)):
        active = np.flatnonzero(data.valid[row])
        order = rng.permutation(len(active))
        target_mass[row, active] = data.target_mass[row, active[order]]
        target_shift[row, active] = data.target_shift[row, active[order]]
    return PathData(
        query_ids=data.query_ids,
        query_node=data.query_node,
        delay=data.delay,
        feature=data.feature,
        prior=data.prior,
        valid=data.valid,
        anchor=data.anchor,
        target_mass=target_mass,
        target_shift=target_shift,
        target_gate=data.target_gate,
        covered_mass=data.covered_mass,
    )


def inference_batch(
    data: PathData, rows: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.asarray(getattr(data, key)[rows])).to(device)
        for key in ("query_node", "delay", "feature", "prior", "valid", "anchor")
    }


def supervision_batch(
    data: PathData, rows: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        torch.from_numpy(np.asarray(getattr(data, key)[rows])).to(device)
        for key in ("target_mass", "target_shift", "target_gate", "covered_mass")
    )


def supervised_loss(
    prediction: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    truth: torch.Tensor,
    target_mass: torch.Tensor,
    target_shift: torch.Tensor,
    target_gate: torch.Tensor,
    covered_mass: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    spectrum = v366.loss_function(prediction, truth)
    sample_weight = covered_mass.clamp(0.0, 1.0)
    sample_weight /= sample_weight.mean().clamp_min(1e-6)
    mass_ce = -torch.sum(
        target_mass * torch.log(diagnostics["energy"].clamp_min(1e-12)), dim=1
    )
    mass_ce = torch.mean(sample_weight * mass_ce)
    shift_ce = -torch.sum(
        target_mass[:, :, None]
        * target_shift
        * torch.log(diagnostics["shift_probability"].clamp_min(1e-12)),
        dim=(1, 2),
    )
    shift_ce = torch.mean(sample_weight * shift_ce)
    gate_mse = torch.mean(
        sample_weight * torch.square(diagnostics["gate"] - target_gate)
    )
    total = (
        spectrum
        + MASS_LOSS_WEIGHT * mass_ce
        + SHIFT_LOSS_WEIGHT * shift_ce
        + GATE_LOSS_WEIGHT * gate_mse
    )
    return total, {
        "spectrum": spectrum,
        "mass_ce": mass_ce,
        "shift_ce": shift_ce,
        "gate_mse": gate_mse,
    }


@torch.no_grad()
def predict(
    model: v366.EikonalSurfaceModel,
    data: PathData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty((len(data.query_ids), v366.N_DELAY), dtype=np.float32)
    gate = np.empty(len(data.query_ids), dtype=np.float32)
    energy = np.empty((len(data.query_ids), v366.HYPOTHESES), dtype=np.float32)
    for start in range(0, len(data.query_ids), TRAIN_BATCH):
        end = min(start + TRAIN_BATCH, len(data.query_ids))
        block, diagnostics = model(
            **inference_batch(data, np.arange(start, end), device)
        )
        output[start:end] = block.cpu().numpy()
        gate[start:end] = diagnostics["gate"].cpu().numpy()
        energy[start:end] = diagnostics["energy"].cpu().numpy()
    return output, gate, energy


def fit_model(
    train: PathData,
    validation: PathData | None,
    truth_d: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[v366.EikonalSurfaceModel, dict[str, object]]:
    v366.set_seed(seed)
    model = v366.EikonalSurfaceModel(
        train.query_node.shape[1], train.feature.shape[2]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=2e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    rng = np.random.default_rng(seed)
    best_epoch = epochs
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    if validation is not None:
        best_epoch = 0
        best_score = float(
            np.mean(v365.row_cos(validation.anchor, truth_d[validation.query_ids]))
        )
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train.query_ids))
        loss_rows: dict[str, list[float]] = {
            key: [] for key in ("total", "spectrum", "mass_ce", "shift_ce", "gate_mse")
        }
        for start in range(0, len(order), TRAIN_BATCH):
            rows = order[start : start + TRAIN_BATCH]
            prediction, diagnostics = model(**inference_batch(train, rows, device))
            truth = torch.from_numpy(truth_d[train.query_ids[rows]]).to(device)
            target_mass, target_shift, target_gate, covered_mass = supervision_batch(
                train, rows, device
            )
            loss, parts = supervised_loss(
                prediction,
                diagnostics,
                truth,
                target_mass,
                target_shift,
                target_gate,
                covered_mass,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_rows["total"].append(float(loss.detach().cpu()))
            for key, value in parts.items():
                loss_rows[key].append(float(value.detach().cpu()))
        scheduler.step()
        if validation is not None and (
            epoch % EVAL_EVERY == 0 or epoch == epochs
        ):
            validation_prediction, validation_gate, validation_energy = predict(
                model, validation, device
            )
            score = float(
                np.mean(
                    v365.row_cos(
                        validation_prediction, truth_d[validation.query_ids]
                    )
                )
            )
            item: dict[str, object] = {
                "epoch": epoch,
                **{key: float(np.mean(value)) for key, value in loss_rows.items()},
                "validation_pdp": score,
                "validation_gate": float(np.mean(validation_gate)),
                "validation_path_ap": v366.surface_ap(
                    validation_energy,
                    v366.ModelData(
                        validation.query_ids,
                        validation.query_node,
                        validation.delay,
                        validation.feature,
                        validation.prior,
                        validation.valid,
                        validation.anchor,
                    ),
                    COMPONENT,
                ),
            }
            history.append(item)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if epoch % 12 == 0 or epoch == epochs:
            print(
                f"v367 seed={seed} epoch={epoch}/{epochs} "
                f"loss={np.mean(loss_rows['total']):.6f} "
                f"mass={np.mean(loss_rows['mass_ce']):.4f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_validation_pdp": (
            float(best_score)
            if validation is not None and np.isfinite(best_score)
            else None
        ),
        "history": history,
    }


COMPONENT: dict[str, np.ndarray] = {}


def run_folds(
    folds: list[np.ndarray],
    valid_ids: np.ndarray,
    truth_d: np.ndarray,
    geometry: tuple,
    node_raw: np.ndarray,
    component: dict[str, np.ndarray],
    regimes: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    all_xy = geometry[0][:, :2]
    fold_rows: list[dict[str, object]] = []
    oof: dict[str, list[np.ndarray]] = {
        "ids": [], "truth": [], "baseline": [], "shuffled": [], "full": []
    }
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(folds[(fold + 1) % len(folds)], fit_ids)
        inner_fit = np.setdiff1d(fit_ids, inner_ids)
        inner_train_set, inner_val_set, inner_node = v366.raw_sets(
            inner_fit, inner_ids, inner_fit, geometry, node_raw
        )
        outer_train_set, outer_val_set, outer_node = v366.raw_sets(
            fit_ids, outer_ids, fit_ids, geometry, node_raw
        )
        inner_train_h = v366.build_hypotheses(inner_train_set, all_xy, component)
        inner_val_h = v366.build_hypotheses(inner_val_set, all_xy, component)
        outer_train_h = v366.build_hypotheses(outer_train_set, all_xy, component)
        outer_val_h = v366.build_hypotheses(outer_val_set, all_xy, component)
        inner_train = make_path_data(
            inner_train_h,
            inner_node[inner_train_h.query_ids],
            truth_d,
            component,
            True,
        )
        inner_val = make_path_data(
            inner_val_h,
            inner_node[inner_val_h.query_ids],
            truth_d,
            component,
            False,
        )
        outer_train = make_path_data(
            outer_train_h,
            outer_node[outer_train_h.query_ids],
            truth_d,
            component,
            True,
        )
        outer_val = make_path_data(
            outer_val_h,
            outer_node[outer_val_h.query_ids],
            truth_d,
            component,
            False,
        )
        modes = {
            "full": (inner_train, outer_train),
            "shuffled": (
                shuffled_supervision(inner_train, 367_000 + fold),
                shuffled_supervision(outer_train, 368_000 + fold),
            ),
        }
        predictions: dict[str, np.ndarray] = {}
        energies: dict[str, np.ndarray] = {}
        training_report: dict[str, object] = {}
        for mode, (inner_train_mode, outer_train_mode) in modes.items():
            seed = 3670 + fold * 10
            inner_model, inner_report = fit_model(
                inner_train_mode,
                inner_val,
                truth_d,
                seed,
                MAX_EPOCHS,
                device,
            )
            selected_epoch = int(inner_report["best_epoch"])
            del inner_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if selected_epoch == 0:
                prediction = outer_val.anchor.copy()
                gate = np.zeros(len(outer_ids), dtype=np.float32)
                energy = outer_val.prior.copy()
                outer_report: dict[str, object] = {"best_epoch": 0}
            else:
                model, outer_report = fit_model(
                    outer_train_mode,
                    None,
                    truth_d,
                    4670 + fold * 10,
                    selected_epoch,
                    device,
                )
                prediction, gate, energy = predict(model, outer_val, device)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            predictions[mode] = prediction
            energies[mode] = energy
            training_report[mode] = {
                "selected_epoch": selected_epoch,
                "inner": inner_report,
                "outer": outer_report,
                "gate": summarize(gate),
            }

        truth = truth_d[outer_ids]
        baseline_metric = v365.evaluate(
            outer_val.anchor, truth, outer_ids, component
        )
        shuffled_metric = v365.evaluate(
            predictions["shuffled"], truth, outer_ids, component
        )
        full_metric = v365.evaluate(
            predictions["full"], truth, outer_ids, component
        )
        outer_model_data = v366.ModelData(
            outer_val.query_ids,
            outer_val.query_node,
            outer_val.delay,
            outer_val.feature,
            outer_val.prior,
            outer_val.valid,
            outer_val.anchor,
        )
        regime_rows = {}
        for regime_name in ("clear", "light", "medium", "heavy"):
            mask = regimes[outer_ids] == regime_name
            if np.any(mask):
                regime_rows[regime_name] = {
                    "count": int(np.sum(mask)),
                    "cosine_lift": float(
                        np.mean(
                            full_metric["cosine_rows"][mask]
                            - baseline_metric["cosine_rows"][mask]
                        )
                    ),
                }
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "outer_count": int(len(outer_ids)),
                "train_supervision": {
                    "covered_mass": summarize(outer_train.covered_mass),
                    "target_gate": summarize(outer_train.target_gate),
                    "active_target_paths": summarize(
                        np.sum(outer_train.target_mass > 0.0, axis=1)
                    ),
                },
                "baseline": v365.metric_summary(baseline_metric),
                "shuffled_labels": v365.metric_summary(shuffled_metric),
                "full": v365.metric_summary(full_metric),
                "full_delta": {
                    "cosine": float(
                        full_metric["cosine"] - baseline_metric["cosine"]
                    ),
                    "top_path_mass_recall": float(
                        full_metric["top_path_mass_recall"]
                        - baseline_metric["top_path_mass_recall"]
                    ),
                    "delay_emd": float(
                        full_metric["delay_emd"] - baseline_metric["delay_emd"]
                    ),
                },
                "full_vs_shuffled_cosine": float(
                    full_metric["cosine"] - shuffled_metric["cosine"]
                ),
                "path_existence_ap": {
                    "prior": v366.surface_ap(
                        outer_val.prior, outer_model_data, component
                    ),
                    "shuffled": v366.surface_ap(
                        energies["shuffled"], outer_model_data, component
                    ),
                    "full": v366.surface_ap(
                        energies["full"], outer_model_data, component
                    ),
                },
                "regimes": regime_rows,
                "training": training_report,
                "leakage_assertions": {
                    "outer_in_path_supervision": 0,
                    "outer_in_donor_bank": 0,
                },
            }
        )
        oof["ids"].append(outer_ids)
        oof["truth"].append(truth)
        oof["baseline"].append(outer_val.anchor)
        oof["shuffled"].append(predictions["shuffled"])
        oof["full"].append(predictions["full"])
        print(
            f"v367 fold={fold} base={baseline_metric['cosine']:.6f} "
            f"shuffle={shuffled_metric['cosine']:.6f} "
            f"full={full_metric['cosine']:.6f}",
            flush=True,
        )
    merged = {key: np.concatenate(value) for key, value in oof.items()}
    order = np.argsort(merged["ids"])
    for key in merged:
        merged[key] = merged[key][order]
    np.savez_compressed(OOF_OUT, **merged)
    return fold_rows, merged


def main() -> None:
    global COMPONENT
    for path in (
        TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
        v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    target = np.load(TARGET_CACHE)
    truth_d = v365.normalize_rows(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    component_cache = np.load(v297.COMPONENT_CACHE)
    COMPONENT = {key: component_cache[key] for key in component_cache.files}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    geometry = v274.load_geometry()
    all_pos, signatures, blocker_count, scene_features = geometry[:4]
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    propagation_geometry = np.load(v238.GEOM_CACHE)
    regimes = regime_of(
        propagation_geometry["train_blocked"].astype(np.float32)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows, merged = run_folds(
        folds,
        valid_ids,
        truth_d,
        geometry,
        node_raw,
        COMPONENT,
        regimes,
        device,
    )
    base_metric = v365.evaluate(
        merged["baseline"], merged["truth"], merged["ids"], COMPONENT
    )
    shuffled_metric = v365.evaluate(
        merged["shuffled"], merged["truth"], merged["ids"], COMPONENT
    )
    full_metric = v365.evaluate(
        merged["full"], merged["truth"], merged["ids"], COMPONENT
    )
    cosine_lift = float(full_metric["cosine"] - base_metric["cosine"])
    recall_lift = float(
        full_metric["top_path_mass_recall"]
        - base_metric["top_path_mass_recall"]
    )
    emd_reduction = float(
        (base_metric["delay_emd"] - full_metric["delay_emd"])
        / max(float(base_metric["delay_emd"]), 1e-20)
    )
    fold_wins = int(
        sum(row["full_delta"]["cosine"] > 0.0 for row in fold_rows)
    )
    full_vs_shuffled = float(
        full_metric["cosine"] - shuffled_metric["cosine"]
    )
    regime_summary = {}
    for regime_name in ("clear", "light", "medium", "heavy"):
        mask = regimes[merged["ids"]] == regime_name
        regime_summary[regime_name] = {
            "count": int(np.sum(mask)),
            "cosine_lift": float(
                np.mean(
                    full_metric["cosine_rows"][mask]
                    - base_metric["cosine_rows"][mask]
                )
            ),
        }
    passed = bool(
        cosine_lift >= MIN_COSINE_LIFT
        and recall_lift >= MIN_TOP_RECALL_LIFT
        and emd_reduction >= MIN_EMD_REDUCTION
        and fold_wins >= MIN_FOLD_WINS
        and full_vs_shuffled >= MIN_PATH_SUPERVISION_LIFT
        and regime_summary["medium"]["cosine_lift"] > 0.0
        and regime_summary["heavy"]["cosine_lift"] > 0.0
        and regime_summary["clear"]["cosine_lift"] >= -MAX_CLEAR_DEGRADATION
    )
    summary = {
        "baseline": v365.metric_summary(base_metric),
        "shuffled_labels": v365.metric_summary(shuffled_metric),
        "full": v365.metric_summary(full_metric),
        "cosine_lift": cosine_lift,
        "top_path_mass_recall_lift": recall_lift,
        "delay_emd_reduction_fraction": emd_reduction,
        "fold_wins": fold_wins,
        "full_vs_shuffled_cosine": full_vs_shuffled,
        "regimes": regime_summary,
    }
    report = {
        "method": VERSION,
        "status": (
            "strict_pdp_module_passed" if passed else "analysis_only_no_candidate"
        ),
        "scope": (
            "normalized global D only; no full-H, phase, total power, Sionna target, or ZIP"
        ),
        "primary_hypothesis": (
            "The v366 eikonal candidate set has a large oracle ceiling, but indirect "
            "spectrum loss reduces path AP. Explicit fit-only component mass, delay "
            "residual, and oracle-action supervision should make path scoring deployable."
        ),
        "compliance": {
            "uses": [
                "official train H continuous components and normalized PDP",
                "official train/test planar coordinates and point-cloud node features",
                "fit-query component-to-surface matching as auxiliary AI supervision",
            ],
            "does_not_use": [
                "outer-fold component labels in training or features",
                "Sionna/synthetic labels or v212 teacher",
                "global source identity, hidden truth, or online feedback",
                "full-H projection or submission packaging",
            ],
        },
        "parameters": {
            "representation": "frozen v366 local eikonal surfaces",
            "mass_loss_weight": MASS_LOSS_WEIGHT,
            "shift_loss_weight": SHIFT_LOSS_WEIGHT,
            "gate_loss_weight": GATE_LOSS_WEIGHT,
            "shift_sigma_bin": SHIFT_SIGMA,
            "max_epochs": MAX_EPOCHS,
            "folds": "v315 five mutually exclusive isolation-stratified spatial folds",
            "control": (
                "identical input/model/seed with component mass and shift labels "
                "shuffled among valid fit-query surfaces"
            ),
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "cosine_lift": f">={MIN_COSINE_LIFT}",
            "top_path_mass_recall_lift": f">={MIN_TOP_RECALL_LIFT}",
            "delay_emd_reduction_fraction": f">={MIN_EMD_REDUCTION}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "full_vs_shuffled_cosine": f">={MIN_PATH_SUPERVISION_LIFT}",
            "medium_heavy": "both positive",
            "clear": f">=-{MAX_CLEAR_DEGRADATION}",
            "passed": passed,
        },
        "outputs": {
            "oof": PATHS.relative(OOF_OUT),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_rule"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
