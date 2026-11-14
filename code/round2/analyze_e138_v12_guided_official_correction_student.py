#!/usr/bin/env python3
"""Learn V12-guided spectral corrections using only deployable official features."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analyze_e86_surface_endpoint_v11_blend import endpoint_target
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META, statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E138-v12-guided-official-correction-student"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v12_guided_official_correction_student"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_v12_guided_official_correction_student"
)
REPORT = ANALYSIS_DIR / "v12_guided_official_correction_student.json"
METRICS = ARTIFACT_DIR / "v12_guided_official_correction_student_metrics.npz"

V7_CHANNEL = (
    PATHS.artifacts / "round2" / "full_sionna_migration" / "oof_v7_nested_channel.npy"
)
V12_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_frozen_geometry_sequence_oof_channel.npy"
)
TEACHER_PAS = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_geometry_sequence_rendered_pas_f16.npy"
)
TEACHER_PDP = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_geometry_sequence_rendered_pdp_f16.npy"
)
OFFICIAL_ANCHOR_PAS = (
    PATHS.artifacts / "round2" / "v3_anchored_neural_rbf" / "oof_pas_neural_f16.npy"
)
OFFICIAL_ANCHOR_PDP = (
    PATHS.artifacts / "round2" / "v3_anchored_neural_rbf" / "oof_pdp_neural_f16.npy"
)
GEOMETRY = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_geometry_visibility_features.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PAS_RANK = 32
PDP_RANK = 32
BETA = 0.20
CYCLES = 4
ORDER = "pas_pdp"
MINIMUM_GAIN = 0.002
MINIMUM_POSITIVE_FOLDS = 4
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trees", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20_260_809_138)
    return parser.parse_args()


def normalize_probability(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, TINY),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def compact_channel_endpoints(
    channel: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(channel)
    pas = np.empty((count, MP, MH, MV, N_UE), dtype=np.float32)
    pdp = np.empty((count, MP, N_UE, S_SC), dtype=np.float32)
    log_power = np.empty(count, dtype=np.float32)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        value = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            full_pas = pas_distribution(value)
            full_pdp = pdp_distribution(value).reshape(
                end - start, MP, MH, MV, N_UE, S_SC
            )
            pas[start:end] = full_pas.mean(dim=-1).cpu().numpy().astype(np.float32)
            pdp[start:end] = (
                full_pdp.mean(dim=(2, 3)).cpu().numpy().astype(np.float32)
            )
            power = value.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
            log_power[start:end] = torch.log10(power.clamp_min(TINY)).cpu().numpy()
    return (
        normalize_probability(pas, (1, 2, 3)),
        normalize_probability(pdp, (3,)),
        log_power,
    )


def compact_cached_endpoints(
    pas_path,
    pdp_path,
) -> tuple[np.ndarray, np.ndarray]:
    pas_source = np.load(pas_path, mmap_mode="r")
    pdp_source = np.load(pdp_path, mmap_mode="r")
    if pas_source.shape == (500, MP, MH, MV, N_UE):
        pas = np.asarray(pas_source, dtype=np.float32)
    elif pas_source.shape == (500, MP, MH, MV, N_UE, S_SC):
        pas = np.asarray(pas_source, dtype=np.float32).mean(axis=-1)
    elif pas_source.shape == (500, MP * MH * MV, N_UE, S_SC):
        pas = np.asarray(pas_source, dtype=np.float32).reshape(
            500, MP, MH, MV, N_UE, S_SC
        ).mean(axis=-1)
    else:
        raise ValueError(f"unexpected PAS endpoint shape: {pas_source.shape}")
    if pdp_source.shape == (500, MP, N_UE, S_SC):
        pdp = np.asarray(pdp_source, dtype=np.float32)
    elif pdp_source.shape == (500, MP * MH * MV, N_UE, S_SC):
        pdp = np.asarray(pdp_source, dtype=np.float32).reshape(
            500, MP, MH, MV, N_UE, S_SC
        ).mean(axis=(2, 3))
    else:
        raise ValueError(f"unexpected PDP endpoint shape: {pdp_source.shape}")
    return (
        normalize_probability(pas, (1, 2, 3)),
        normalize_probability(pdp, (3,)),
    )


def fit_basis(delta: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    matrix = np.asarray(delta, dtype=np.float64).reshape(len(delta), -1)
    _, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
    keep = min(rank, len(vectors))
    basis = vectors[:keep].astype(np.float32)
    total = float(np.sum(singular * singular, dtype=np.float64))
    explained = float(
        np.sum(singular[:keep] * singular[:keep], dtype=np.float64)
        / max(total, TINY)
    )
    return basis, explained


def project_coefficients(delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    flat = np.asarray(delta, dtype=np.float32).reshape(len(delta), -1)
    return (flat @ basis.T).astype(np.float32)


def reconstruct_endpoint(
    base: np.ndarray,
    coefficients: np.ndarray,
    basis: np.ndarray,
    probability_axes: tuple[int, ...],
) -> np.ndarray:
    amplitude = np.sqrt(np.maximum(base, 0.0)).reshape(len(base), -1)
    amplitude = amplitude + np.asarray(coefficients, dtype=np.float32) @ basis
    power = np.maximum(amplitude, 0.0).reshape(base.shape) ** 2
    return normalize_probability(power, probability_axes)


def circular_moment(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = probability.shape[-1]
    phase = np.exp(2j * np.pi * np.arange(bins, dtype=np.float64) / bins)
    moment = np.sum(probability * phase, axis=-1, dtype=np.complex128)
    return moment.real.astype(np.float32), moment.imag.astype(np.float32)


def endpoint_summary(pas: np.ndarray, pdp: np.ndarray) -> np.ndarray:
    pas_group = np.transpose(pas, (0, 4, 1, 2, 3))
    pas_flat = pas_group.reshape(len(pas), N_UE, -1)
    pas_entropy = -np.sum(
        pas_flat * np.log(np.maximum(pas_flat, 1e-12)), axis=-1
    ) / np.log(pas_flat.shape[-1])
    pas_max = np.max(pas_flat, axis=-1)
    pas_h = pas_group.sum(axis=(2, 4), dtype=np.float64).astype(np.float32)
    pas_v = pas_group.sum(axis=(2, 3), dtype=np.float64).astype(np.float32)
    pas_pol = pas_group.sum(axis=(3, 4), dtype=np.float64).astype(np.float32)
    h_real, h_imag = circular_moment(pas_h)
    v_real, v_imag = circular_moment(pas_v)

    pdp_group = pdp.reshape(len(pdp), MP * N_UE, S_SC)
    pdp_entropy = -np.sum(
        pdp_group * np.log(np.maximum(pdp_group, 1e-12)), axis=-1
    ) / np.log(S_SC)
    pdp_max = np.max(pdp_group, axis=-1)
    pdp_real, pdp_imag = circular_moment(pdp_group)
    pdp_top = np.argpartition(pdp_group, -4, axis=-1)[..., -4:]
    pdp_top_mass = np.take_along_axis(pdp_group, pdp_top, axis=-1).sum(axis=-1)

    blocks = [
        pas_entropy,
        pas_max,
        pas_pol.reshape(len(pas), -1),
        h_real,
        h_imag,
        v_real,
        v_imag,
        pdp_entropy,
        pdp_max,
        pdp_real,
        pdp_imag,
        pdp_top_mass,
    ]
    return np.concatenate([value.reshape(len(pas), -1) for value in blocks], axis=1).astype(
        np.float32
    )


def endpoint_comparison(
    base_pas: np.ndarray,
    base_pdp: np.ndarray,
    anchor_pas: np.ndarray,
    anchor_pdp: np.ndarray,
) -> np.ndarray:
    pas_base = np.transpose(base_pas, (0, 4, 1, 2, 3)).reshape(
        len(base_pas), N_UE, -1
    )
    pas_anchor = np.transpose(anchor_pas, (0, 4, 1, 2, 3)).reshape(
        len(base_pas), N_UE, -1
    )
    pdp_base = base_pdp.reshape(len(base_pdp), MP * N_UE, S_SC)
    pdp_anchor = anchor_pdp.reshape(len(base_pdp), MP * N_UE, S_SC)

    def pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        cosine = np.sum(left * right, axis=-1) / np.maximum(
            np.sqrt(np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1)),
            TINY,
        )
        hellinger = np.sqrt(
            np.maximum(
                0.0,
                1.0 - np.sum(np.sqrt(np.maximum(left * right, 0.0)), axis=-1),
            )
        )
        return np.concatenate((cosine, hellinger), axis=1)

    return np.concatenate((pair(pas_base, pas_anchor), pair(pdp_base, pdp_anchor)), axis=1).astype(
        np.float32
    )


def spatial_features(query_ids: np.ndarray, site: np.ndarray) -> np.ndarray:
    setup, train_positions, _, _, _ = load_positions()
    point = train_positions[query_ids].astype(np.float32)
    bs = setup.bs_positions[site].astype(np.float32)
    delta = point - bs
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    azimuth = np.arctan2(delta[:, 1], delta[:, 0])[:, None]
    values = [
        point[:, :2] / 200.0,
        delta[:, :2] / 200.0,
        distance / 200.0,
        np.sin(azimuth),
        np.cos(azimuth),
        np.eye(2, dtype=np.float32)[site],
    ]
    for scale in (4.0, 8.0, 16.0, 32.0, 64.0):
        phase = 2.0 * np.pi * point[:, :2] / scale
        values.extend((np.sin(phase), np.cos(phase)))
    return np.concatenate(values, axis=1).astype(np.float32)


def build_features(
    query_ids: np.ndarray,
    site: np.ndarray,
    geometry: np.ndarray,
    base_pas: np.ndarray,
    base_pdp: np.ndarray,
    anchor_pas: np.ndarray,
    anchor_pdp: np.ndarray,
    base_log_power: np.ndarray,
    pas_basis: np.ndarray,
    pdp_basis: np.ndarray,
) -> np.ndarray:
    anchor_pas_delta = np.sqrt(np.maximum(anchor_pas, 0.0)) - np.sqrt(
        np.maximum(base_pas, 0.0)
    )
    anchor_pdp_delta = np.sqrt(np.maximum(anchor_pdp, 0.0)) - np.sqrt(
        np.maximum(base_pdp, 0.0)
    )
    blocks = [
        spatial_features(query_ids, site),
        geometry[query_ids],
        base_log_power[:, None] / 10.0,
        endpoint_summary(base_pas, base_pdp),
        endpoint_summary(anchor_pas, anchor_pdp),
        endpoint_comparison(base_pas, base_pdp, anchor_pas, anchor_pdp),
        project_coefficients(anchor_pas_delta, pas_basis),
        project_coefficients(anchor_pdp_delta, pdp_basis),
    ]
    output = np.concatenate(blocks, axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("non-finite student features")
    return output


def normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    error = np.mean((prediction - target) ** 2, axis=0)
    scale = np.var(target, axis=0) + 1e-8
    return float(np.mean(error / scale))


def fit_predict_models(
    features: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    held_rows: np.ndarray,
    fold_id: np.ndarray,
    trees: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    inner_predictions: dict[str, np.ndarray] = {
        name: np.zeros((len(train_rows), target.shape[1]), dtype=np.float32)
        for name in ("ridge", "trees", "blend")
    }
    train_fold = fold_id[train_rows]
    for inner in np.unique(train_fold):
        fit_local = train_rows[train_fold != inner]
        valid_mask = train_fold == inner
        valid_local = train_rows[valid_mask]
        scaler = StandardScaler().fit(features[fit_local])
        ridge = Ridge(alpha=10.0).fit(
            scaler.transform(features[fit_local]), target[fit_local]
        )
        ridge_prediction = ridge.predict(scaler.transform(features[valid_local])).astype(
            np.float32
        )
        tree = ExtraTreesRegressor(
            n_estimators=max(96, trees // 2),
            max_depth=16,
            min_samples_leaf=6,
            max_features=0.70,
            bootstrap=False,
            n_jobs=-1,
            random_state=(seed + int(inner) * 101) % (2**32),
        ).fit(features[fit_local], target[fit_local])
        tree_prediction = tree.predict(features[valid_local]).astype(np.float32)
        inner_predictions["ridge"][valid_mask] = ridge_prediction
        inner_predictions["trees"][valid_mask] = tree_prediction
        inner_predictions["blend"][valid_mask] = 0.5 * (
            ridge_prediction + tree_prediction
        )

    scores = {
        name: normalized_mse(value, target[train_rows])
        for name, value in inner_predictions.items()
    }
    selected = min(scores, key=scores.get)
    scaler = StandardScaler().fit(features[train_rows])
    ridge = Ridge(alpha=10.0).fit(
        scaler.transform(features[train_rows]), target[train_rows]
    )
    ridge_prediction = ridge.predict(scaler.transform(features[held_rows])).astype(
        np.float32
    )
    tree = ExtraTreesRegressor(
        n_estimators=trees,
        max_depth=16,
        min_samples_leaf=6,
        max_features=0.70,
        bootstrap=False,
        n_jobs=-1,
        random_state=seed % (2**32),
    ).fit(features[train_rows], target[train_rows])
    tree_prediction = tree.predict(features[held_rows]).astype(np.float32)
    candidates = {
        "ridge": ridge_prediction,
        "trees": tree_prediction,
        "blend": 0.5 * (ridge_prediction + tree_prediction),
    }
    prediction = candidates[selected]
    lower = np.quantile(target[train_rows], 0.01, axis=0)
    upper = np.quantile(target[train_rows], 0.99, axis=0)
    prediction = np.clip(prediction, lower, upper).astype(np.float32)
    return prediction, {
        "selected": selected,
        "inner_normalized_mse": scores,
        "train_count": int(len(train_rows)),
        "held_count": int(len(held_rows)),
    }


def expanded_endpoint(
    pas: np.ndarray, pdp: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pas_tensor = torch.as_tensor(pas, dtype=torch.float32, device=device)
    pas_tensor = pas_tensor[..., None].expand(-1, -1, -1, -1, -1, S_SC)
    pdp_tensor = torch.as_tensor(pdp, dtype=torch.float32, device=device)
    pdp_tensor = (
        pdp_tensor[:, :, None, :, :]
        .expand(-1, -1, MH * MV, -1, -1)
        .reshape(-1, MP * MH * MV, N_UE, S_SC)
    )
    return pas_tensor, pdp_tensor


def evaluate_method(
    base_path,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    endpoint_pas: np.ndarray,
    endpoint_pdp: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_all = np.load(base_path, mmap_mode="r")
    values = [np.empty(len(query_ids), dtype=np.float64) for _ in range(4)]
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        base = torch.as_tensor(
            np.array(base_all[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        full_endpoint_pas, full_endpoint_pdp = expanded_endpoint(
            endpoint_pas[start:end], endpoint_pdp[start:end], device
        )
        with torch.inference_mode():
            truth_pas = pas_distribution(truth)
            truth_pdp = pdp_distribution(truth)
            base_pas = pas_distribution(base)
            base_pdp = pdp_distribution(base)
            desired_pas = endpoint_target(
                base_pas, full_endpoint_pas, BETA, "sqrt", (1, 2, 3)
            )
            desired_pdp = endpoint_target(
                base_pdp, full_endpoint_pdp, BETA, "sqrt", (3,)
            )
            prediction = project_joint(base, desired_pas, desired_pdp, CYCLES, ORDER)
            point = point_metrics(prediction, truth, truth_pas, truth_pdp)
        for destination, source in zip(values, point):
            destination[start:end] = source
    return tuple(values)


def baseline_metrics(
    channel_path,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel = np.load(channel_path, mmap_mode="r")
    values = [np.empty(len(query_ids), dtype=np.float64) for _ in range(4)]
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        prediction = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        with torch.inference_mode():
            point = point_metrics(
                prediction, truth, pas_distribution(truth), pdp_distribution(truth)
            )
        for destination, source in zip(values, point):
            destination[start:end] = source
    return tuple(values)


def summarize_method(
    values: tuple[np.ndarray, ...],
    baseline: tuple[np.ndarray, ...],
    fold_id: np.ndarray,
) -> dict[str, object]:
    overall = statistics(*values)
    anchor = statistics(*baseline)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        current = statistics(*values, mask)
        base = statistics(*baseline, mask)
        folds.append(
            {
                "fold": fold,
                "metrics": current,
                "baseline": base,
                "delta": float(current["score"] - base["score"]),
            }
        )
    return {
        "metrics": overall,
        "delta_to_v7": float(overall["score"] - anchor["score"]),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "worst_fold_delta": float(min(row["delta"] for row in folds)),
        "folds": folds,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    geometry = np.load(GEOMETRY, allow_pickle=False)["train"].astype(np.float32)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")

    v7_pas, v7_pdp, v7_log_power = compact_channel_endpoints(
        np.load(V7_CHANNEL, mmap_mode="r"), device, args.batch_size
    )
    truth_pas, truth_pdp, _ = compact_channel_endpoints(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device, args.batch_size
    )
    anchor_pas, anchor_pdp = compact_cached_endpoints(
        OFFICIAL_ANCHOR_PAS, OFFICIAL_ANCHOR_PDP
    )
    teacher_pas, teacher_pdp = compact_cached_endpoints(TEACHER_PAS, TEACHER_PDP)

    student_pas = np.empty_like(v7_pas)
    student_pdp = np.empty_like(v7_pdp)
    oracle_pas = np.empty_like(v7_pas)
    oracle_pdp = np.empty_like(v7_pdp)
    anchor_projection_pas = np.empty_like(v7_pas)
    anchor_projection_pdp = np.empty_like(v7_pdp)
    fold_records: list[dict[str, object]] = []

    v7_pas_amp = np.sqrt(np.maximum(v7_pas, 0.0))
    v7_pdp_amp = np.sqrt(np.maximum(v7_pdp, 0.0))
    truth_pas_delta = np.sqrt(np.maximum(truth_pas, 0.0)) - v7_pas_amp
    truth_pdp_delta = np.sqrt(np.maximum(truth_pdp, 0.0)) - v7_pdp_amp
    teacher_pas_delta = np.sqrt(np.maximum(teacher_pas, 0.0)) - v7_pas_amp
    teacher_pdp_delta = np.sqrt(np.maximum(teacher_pdp, 0.0)) - v7_pdp_amp
    anchor_pas_delta = np.sqrt(np.maximum(anchor_pas, 0.0)) - v7_pas_amp
    anchor_pdp_delta = np.sqrt(np.maximum(anchor_pdp, 0.0)) - v7_pdp_amp

    for outer in range(5):
        train_rows = np.flatnonzero(fold_id != outer)
        held_rows = np.flatnonzero(fold_id == outer)
        pas_basis, pas_explained = fit_basis(teacher_pas_delta[train_rows], PAS_RANK)
        pdp_basis, pdp_explained = fit_basis(teacher_pdp_delta[train_rows], PDP_RANK)
        features = build_features(
            query_ids,
            site,
            geometry,
            v7_pas,
            v7_pdp,
            anchor_pas,
            anchor_pdp,
            v7_log_power,
            pas_basis,
            pdp_basis,
        )
        pas_target = project_coefficients(truth_pas_delta, pas_basis)
        pdp_target = project_coefficients(truth_pdp_delta, pdp_basis)
        target = np.concatenate((pas_target, pdp_target), axis=1)
        prediction, model_record = fit_predict_models(
            features,
            target,
            train_rows,
            held_rows,
            fold_id,
            args.trees,
            (args.seed + outer * 1009) % (2**32),
        )
        pas_prediction = prediction[:, : len(pas_basis)]
        pdp_prediction = prediction[:, len(pas_basis) :]
        student_pas[held_rows] = reconstruct_endpoint(
            v7_pas[held_rows], pas_prediction, pas_basis, (1, 2, 3)
        )
        student_pdp[held_rows] = reconstruct_endpoint(
            v7_pdp[held_rows], pdp_prediction, pdp_basis, (3,)
        )
        oracle_pas[held_rows] = reconstruct_endpoint(
            v7_pas[held_rows], pas_target[held_rows], pas_basis, (1, 2, 3)
        )
        oracle_pdp[held_rows] = reconstruct_endpoint(
            v7_pdp[held_rows], pdp_target[held_rows], pdp_basis, (3,)
        )
        anchor_projection_pas[held_rows] = reconstruct_endpoint(
            v7_pas[held_rows],
            project_coefficients(anchor_pas_delta[held_rows], pas_basis),
            pas_basis,
            (1, 2, 3),
        )
        anchor_projection_pdp[held_rows] = reconstruct_endpoint(
            v7_pdp[held_rows],
            project_coefficients(anchor_pdp_delta[held_rows], pdp_basis),
            pdp_basis,
            (3,),
        )
        fold_records.append(
            {
                "outer": outer,
                "pas_teacher_basis_explained_energy": pas_explained,
                "pdp_teacher_basis_explained_energy": pdp_explained,
                "feature_count": int(features.shape[1]),
                "model": model_record,
            }
        )
        print(
            f"E138 outer={outer} model={model_record['selected']} "
            f"basis={pas_explained:.4f}/{pdp_explained:.4f}",
            flush=True,
        )

    baseline = baseline_metrics(
        V7_CHANNEL, truth_all, query_ids, device, args.batch_size
    )
    v12_values = baseline_metrics(
        V12_CHANNEL, truth_all, query_ids, device, args.batch_size
    )
    method_endpoints = {
        "official_anchor_projection": (anchor_projection_pas, anchor_projection_pdp),
        "student": (student_pas, student_pdp),
        "teacher_subspace_oracle": (oracle_pas, oracle_pdp),
        "direct_teacher_endpoint": (teacher_pas, teacher_pdp),
    }
    method_values: dict[str, tuple[np.ndarray, ...]] = {}
    summaries: dict[str, object] = {}
    for name, (pas_endpoint, pdp_endpoint) in method_endpoints.items():
        values = evaluate_method(
            V7_CHANNEL,
            truth_all,
            query_ids,
            pas_endpoint,
            pdp_endpoint,
            device,
            args.batch_size,
        )
        method_values[name] = values
        summaries[name] = summarize_method(values, baseline, fold_id)
        print(
            f"E138 {name} score={summaries[name]['metrics']['score']:.6f} "
            f"delta={summaries[name]['delta_to_v7']:+.6f}",
            flush=True,
        )

    baseline_summary = statistics(*baseline)
    v12_summary = statistics(*v12_values)
    primary = summaries["student"]
    gate = {
        "minimum_score_gain": MINIMUM_GAIN,
        "minimum_positive_folds": MINIMUM_POSITIVE_FOLDS,
        "actual_score_gain": float(primary["delta_to_v7"]),
        "actual_positive_folds": int(primary["positive_folds"]),
        "pas_not_worse": bool(
            primary["metrics"]["pas"] >= baseline_summary["pas"] - 1e-4
        ),
        "pdp_not_worse": bool(
            primary["metrics"]["pdp"] >= baseline_summary["pdp"] - 1e-4
        ),
        "nmse_not_worse": bool(
            primary["metrics"]["nmse"] <= baseline_summary["nmse"] + 1e-3
        ),
    }
    gate["passed"] = bool(
        gate["actual_score_gain"] >= MINIMUM_GAIN
        and gate["actual_positive_folds"] >= MINIMUM_POSITIVE_FOLDS
        and gate["pas_not_worse"]
        and gate["pdp_not_worse"]
        and gate["nmse_not_worse"]
    )

    np.savez_compressed(
        METRICS,
        query_ids=query_ids,
        fold_id=fold_id,
        student_endpoint_pas=student_pas.astype(np.float16),
        student_endpoint_pdp=student_pdp.astype(np.float16),
        oracle_endpoint_pas=oracle_pas.astype(np.float16),
        oracle_endpoint_pdp=oracle_pdp.astype(np.float16),
        anchor_projection_endpoint_pas=anchor_projection_pas.astype(np.float16),
        anchor_projection_endpoint_pdp=anchor_projection_pdp.astype(np.float16),
        **{
            f"v7_{key}": value
            for key, value in zip(("pas", "pdp", "error", "energy"), baseline)
        },
        **{
            f"{name}_{key}": value
            for name, values in method_values.items()
            for key, value in zip(("pas", "pdp", "error", "energy"), values)
        },
    )
    payload = {
        "experiment": VERSION,
        "status": "strict_screen_complete",
        "primary_hypothesis": (
            "The score-improving spectral transformations demonstrated by V12 span a "
            "low-dimensional correction space whose coefficients can be predicted from "
            "V7 plus official geometry and official-neighbor spectra, without any held-fold "
            "or test-point Sionna input."
        ),
        "protocol": {
            "canonical_fixed500": True,
            "teacher_role": (
                "V12's official-train surface endpoint defines each outer-fold correction "
                "basis only; official truth defines the coefficient labels."
            ),
            "held_teacher_used_by_student": False,
            "student_inputs": (
                "official geometry, coordinates, V7 official-only carrier descriptors, "
                "and official-neighbor OOF spectral anchors"
            ),
            "projection": {
                "beta": BETA,
                "cycles": CYCLES,
                "order": ORDER,
            },
            "basis_rank": {"pas": PAS_RANK, "pdp": PDP_RANK},
        },
        "baseline_v7": baseline_summary,
        "reference_v12": {
            **v12_summary,
            "delta_to_v7": float(v12_summary["score"] - baseline_summary["score"]),
        },
        "fold_training": fold_records,
        "methods": summaries,
        "gate": gate,
        "decision": {
            "passed": bool(gate["passed"]),
            "generate_candidate": False,
            "reason": (
                "A fixed500 screen cannot be deployed directly; passing would trigger "
                "full-4000 strict base-conditioned training."
                if gate["passed"]
                else "The V12-guided correction student did not pass the complete-score gate."
            ),
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "metrics": PATHS.relative(METRICS),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
