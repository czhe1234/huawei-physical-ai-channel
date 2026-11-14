#!/usr/bin/env python3
"""Cross-fit a deployable predictor for explicit-path PDP energy allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import torch

from analyze_joint_path_token_oracle import FREQUENCIES, OOF_META, REGIME_CACHE, v9_channel
from analyze_joint_path_token_pdp_isolation import evaluate_grid
from analyze_v4_exact_joint_projection import pdp_distribution, truth_views
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E62-explicit-path-pdp-predictor-screen"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n469_s24_k128_b096_d064_scatter1_sel106"
)
ORACLE_METRICS = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_pdp_isolation"
    / "pdp_isolation_metrics_n469.npz"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "explicit_path_pdp_predictor"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_explicit_path_pdp_predictor"
REPORT = ANALYSIS_DIR / "explicit_path_pdp_predictor_screen.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--max-features", type=float, default=0.7)
    parser.add_argument("--target-clip", type=float, default=3.0)
    parser.add_argument("--random-state", type=int, default=106)
    parser.add_argument("--reuse-features", action="store_true")
    return parser.parse_args()


def one_hot(values: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    return np.stack([values == value for value in classes], axis=-1).astype(np.float32)


def angular_hash(values: np.ndarray, moduli: tuple[int, ...]) -> np.ndarray:
    blocks = []
    values = values.astype(np.float64)
    for modulus in moduli:
        phase = 2.0 * np.pi * np.mod(values, modulus) / float(modulus)
        blocks.extend((np.sin(phase)[..., None], np.cos(phase)[..., None]))
    return np.concatenate(blocks, axis=-1).astype(np.float32)


def unit_complex_features(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    unit = np.divide(values, magnitude, out=np.ones_like(values), where=magnitude > 1e-20)
    return np.concatenate((unit.real, unit.imag), axis=-1).astype(np.float32)


def build_features(
    rows: np.ndarray,
    query_ids: np.ndarray,
    regime: np.ndarray,
    output_path: Path,
    path_cache: Path = PATH_CACHE,
    split: str = "train",
    source_classes: tuple[int, ...] = (0, 1),
    presence_seed_count: int = 24,
) -> np.ndarray:
    if split not in ("train", "test"):
        raise ValueError(f"unsupported feature split: {split}")
    if presence_seed_count <= 0:
        raise ValueError("presence_seed_count must be positive")
    mask = np.load(path_cache / "mask.npy", mmap_mode="r")
    tau = np.asarray(np.load(path_cache / "tau.npy", mmap_mode="r"), dtype=np.float32)
    angles = np.asarray(np.load(path_cache / "angles.npy", mmap_mode="r"), dtype=np.float32)
    energy = np.asarray(np.load(path_cache / "energy.npy", mmap_mode="r"), dtype=np.float32)
    fraction = np.asarray(
        np.load(path_cache / "energy_fraction.npy", mmap_mode="r"), dtype=np.float32
    )
    presence = np.asarray(
        np.load(path_cache / "presence.npy", mmap_mode="r"), dtype=np.float32
    )
    source = np.asarray(np.load(path_cache / "source.npy", mmap_mode="r"), dtype=np.int8)
    count = np.asarray(
        np.load(path_cache / "interaction_count.npy", mmap_mode="r"), dtype=np.int64
    )
    interactions = np.asarray(
        np.load(path_cache / "interactions.npy", mmap_mode="r"), dtype=np.uint8
    )
    primitives = np.asarray(
        np.load(path_cache / "primitives.npy", mmap_mode="r"), dtype=np.uint32
    )
    vertices = np.asarray(
        np.load(path_cache / "vertices.npy", mmap_mode="r"), dtype=np.float32
    )
    coefficient = np.load(path_cache / "coefficient.npy", mmap_mode="r")
    n_points, n_paths = mask.shape

    delay = np.mod(tau * 30_000.0 * 192.0, 192.0)
    delay_phase = 2.0 * np.pi * delay / 192.0
    blocks = [
        (delay / 192.0)[..., None].astype(np.float32),
        np.sin(delay_phase)[..., None].astype(np.float32),
        np.cos(delay_phase)[..., None].astype(np.float32),
        np.sin(2.0 * delay_phase)[..., None].astype(np.float32),
        np.cos(2.0 * delay_phase)[..., None].astype(np.float32),
        np.concatenate((np.sin(angles), np.cos(angles)), axis=-1).astype(np.float32),
        np.clip(np.log10(np.maximum(energy, 1e-30)), -30.0, 10.0)[..., None],
        np.clip(np.log10(np.maximum(fraction, 1e-12)), -12.0, 0.0)[..., None],
        np.minimum(presence, float(presence_seed_count))[..., None]
        / float(presence_seed_count),
        np.minimum(count, 16)[..., None].astype(np.float32) / 16.0,
        one_hot(source, source_classes),
    ]

    rank = np.argsort(np.argsort(-fraction, axis=1), axis=1).astype(np.float32)
    blocks.append((rank / max(n_paths - 1, 1))[..., None])
    step = np.arange(16, dtype=np.int64)[None, None, :]
    valid_step = step < count[..., None]
    denominator = np.maximum(count, 1)[..., None].astype(np.float32)
    interaction_hist = np.stack(
        [np.sum((interactions == value) & valid_step, axis=2) for value in (1, 2, 4)],
        axis=-1,
    ).astype(np.float32)
    blocks.append(interaction_hist / denominator)
    first_interaction = interactions[..., 0]
    last_index = np.clip(count - 1, 0, 15)[..., None]
    last_interaction = np.take_along_axis(interactions, last_index, axis=2)[..., 0]
    blocks.extend(
        (
            one_hot(first_interaction, (0, 1, 2, 4)),
            one_hot(last_interaction, (0, 1, 2, 4)),
        )
    )
    first_primitive = primitives[..., 0]
    last_primitive = np.take_along_axis(primitives, last_index, axis=2)[..., 0]
    blocks.extend(
        (
            angular_hash(first_primitive, (97, 389, 1543)),
            angular_hash(last_primitive, (97, 389, 1543)),
        )
    )

    setup, train_positions, test_positions, train_site, test_site = load_positions()
    domain_positions = train_positions if split == "train" else test_positions
    domain_site = train_site if split == "train" else test_site
    point_ids = query_ids[rows]
    positions = domain_positions[point_ids]
    sites = domain_site[point_ids]
    bs = setup.bs_positions[sites, :2]
    relative = positions[:, :2] - bs
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    bearing = np.arctan2(relative[:, 1], relative[:, 0])[:, None]
    center = train_positions[:, :2].mean(axis=0)
    scale = train_positions[:, :2].std(axis=0) + 1e-6
    train_relative = train_positions[:, :2] - setup.bs_positions[train_site, :2]
    relative_scale = np.std(train_relative, axis=0) + 1e-6
    distance_scale = np.median(np.linalg.norm(train_relative, axis=1)) + 1e-6
    point = np.concatenate(
        (
            (positions[:, :2] - center) / scale,
            relative / relative_scale,
            distance / distance_scale,
            np.sin(bearing),
            np.cos(bearing),
            one_hot(sites, (0, 1)),
            one_hot(regime[rows], (0, 1, 2, 3)),
        ),
        axis=1,
    ).astype(np.float32)
    blocks.append(np.broadcast_to(point[:, None, :], (n_points, n_paths, point.shape[1])))

    first_vertex = vertices[..., 0, :2]
    gather_index = np.repeat(last_index[..., None], 2, axis=3)
    last_vertex = np.take_along_axis(vertices[..., :2], gather_index, axis=2)[..., 0, :]
    position_xy = positions[:, None, :2]
    vertex_scale = np.std(train_positions[:, :2], axis=0) + 1e-6
    blocks.extend(
        (
            ((first_vertex - position_xy) / vertex_scale).astype(np.float32),
            ((last_vertex - position_xy) / vertex_scale).astype(np.float32),
            ((last_vertex - first_vertex) / vertex_scale).astype(np.float32),
        )
    )
    segment = np.diff(vertices[..., :2], axis=2)
    segment_valid = np.arange(15)[None, None, :] < np.maximum(count - 1, 0)[..., None]
    route_length = np.sum(np.linalg.norm(segment, axis=-1) * segment_valid, axis=2)
    blocks.append(np.log1p(route_length)[..., None].astype(np.float32))

    scalar_dim = sum(block.shape[-1] for block in blocks)
    coefficient_dim = 57
    features = np.empty((n_points, n_paths, scalar_dim + coefficient_dim), dtype=np.float32)
    features[..., :scalar_dim] = np.concatenate(blocks, axis=-1)
    for start in range(0, n_points, 16):
        end = min(start + 16, n_points)
        shaped = np.asarray(coefficient[start:end]).reshape(
            end - start, n_paths, 2, 16, 8, 4
        )
        power = np.abs(shaped) ** 2
        port_power = power.sum(axis=(3, 4))
        port_fraction = port_power / np.maximum(
            port_power.sum(axis=(2, 3), keepdims=True), 1e-30
        )
        total_log_power = np.log10(
            np.maximum(power.sum(axis=(2, 3, 4, 5)), 1e-30)
        )[..., None]
        reference = shaped[..., 0, 0, :].reshape(end - start, n_paths, 8)
        horizontal = np.sum(
            np.conj(shaped[..., :-1, :, :]) * shaped[..., 1:, :, :], axis=(3, 4)
        ).reshape(end - start, n_paths, 8)
        vertical = np.sum(
            np.conj(shaped[..., :, :-1, :]) * shaped[..., :, 1:, :], axis=(3, 4)
        ).reshape(end - start, n_paths, 8)
        coefficient_features = np.concatenate(
            (
                total_log_power.astype(np.float32),
                port_fraction.reshape(end - start, n_paths, 8).astype(np.float32),
                unit_complex_features(reference),
                unit_complex_features(horizontal),
                unit_complex_features(vertical),
            ),
            axis=-1,
        )
        features[start:end, :, scalar_dim:] = coefficient_features
    features[~np.asarray(mask)] = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    return features


def target_values(amplitudes: np.ndarray, mask: np.ndarray, clip: float) -> np.ndarray:
    target = np.clip(np.log(np.maximum(amplitudes, 1e-8)), -clip, clip)
    for row in range(len(target)):
        valid = mask[row]
        if np.any(valid):
            target[row, valid] -= np.median(target[row, valid])
    return np.clip(target, -clip, clip).astype(np.float32)


def render_pdp(
    amplitudes: np.ndarray,
    coefficient: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    frequency = torch.as_tensor(FREQUENCIES, device=device, dtype=torch.float32)
    blocks = []
    for start in range(0, len(amplitudes), batch_size):
        end = min(start + batch_size, len(amplitudes))
        coefficient_tensor = torch.as_tensor(
            np.asarray(coefficient[start:end]), device=device
        )
        tau_tensor = torch.as_tensor(np.asarray(tau[start:end]), device=device)
        amplitude_tensor = torch.as_tensor(amplitudes[start:end], device=device)
        mask_tensor = torch.as_tensor(np.asarray(mask[start:end]), device=device)
        phase = torch.exp(
            -2j * torch.pi * tau_tensor[:, :, None] * frequency[None, None]
        )
        channel = torch.einsum(
            "bk,bkmu,bks->bmus",
            (amplitude_tensor * mask_tensor).to(coefficient_tensor.dtype),
            coefficient_tensor,
            phase,
        )
        blocks.append(pdp_distribution(channel).cpu().numpy().astype(np.float32))
    return np.concatenate(blocks)


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (PATH_CACHE, ORACLE_METRICS, OOF_META, REGIME_CACHE)
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    oracle = np.load(ORACLE_METRICS, allow_pickle=False)
    rows = oracle["rows"].astype(np.int64)
    amplitudes = oracle["amplitudes"].astype(np.float32)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_all = metadata["fold_id"].astype(np.int8)
    fold_id = fold_all[rows]
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    mask = np.asarray(np.load(PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool)
    feature_path = ARTIFACT_DIR / "path_features_n469.npy"
    if args.reuse_features and feature_path.is_file():
        features = np.load(feature_path, mmap_mode="r")
    else:
        features = build_features(rows, query_ids, regime, feature_path)
    target = target_values(amplitudes, mask, args.target_clip)
    fraction = np.asarray(
        np.load(PATH_CACHE / "energy_fraction.npy", mmap_mode="r"), dtype=np.float32
    )
    normalized_fraction = fraction / np.maximum(fraction.max(axis=1, keepdims=True), 1e-12)
    sample_weight = 0.25 + 0.75 * np.sqrt(np.clip(normalized_fraction, 0.0, 1.0))
    prediction = np.zeros_like(target)
    model_paths = []
    fold_diagnostics = []
    for fold in range(5):
        train_points = fold_id != fold
        held_points = fold_id == fold
        train_mask = mask & train_points[:, None]
        held_mask = mask & held_points[:, None]
        model = ExtraTreesRegressor(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            n_jobs=-1,
            random_state=args.random_state + fold,
        )
        model.fit(
            np.asarray(features)[train_mask],
            target[train_mask],
            sample_weight=sample_weight[train_mask],
        )
        prediction[held_mask] = model.predict(np.asarray(features)[held_mask]).astype(
            np.float32
        )
        model_path = ARTIFACT_DIR / f"fold{fold}_extra_trees.joblib"
        joblib.dump(model, model_path, compress=3)
        model_paths.append(PATHS.relative(model_path))
        fold_diagnostics.append(
            {
                "fold": fold,
                "train_points": int(np.sum(train_points)),
                "held_points": int(np.sum(held_points)),
                "train_tokens": int(np.sum(train_mask)),
                "held_tokens": int(np.sum(held_mask)),
                "held_log_amplitude_mae": float(
                    np.mean(np.abs(prediction[held_mask] - target[held_mask]))
                ),
                "held_log_amplitude_correlation": float(
                    np.corrcoef(prediction[held_mask], target[held_mask])[0, 1]
                ),
            }
        )
        print(f"path PDP predictor fold {fold + 1}/5 complete", flush=True)
    for row in range(len(prediction)):
        valid = mask[row]
        if np.any(valid):
            prediction[row, valid] -= np.median(prediction[row, valid])
    np.save(ARTIFACT_DIR / "oof_log_amplitude_prediction.npy", prediction)

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    coefficient = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
    tau = np.load(PATH_CACHE / "tau.npy", mmap_mode="r")
    base = v9_channel(rows, device, args.batch_size)
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = np.asarray(train_h[query_ids[rows]], dtype=np.complex64)
    grids = []
    all_metrics = {}
    for blend in (0.0, 0.25, 0.50, 0.75, 1.0):
        predicted_amplitude = np.exp(
            np.clip(blend * prediction, -args.target_clip, args.target_clip)
        ).astype(np.float32)
        path_pdp = render_pdp(
            predicted_amplitude,
            coefficient,
            tau,
            mask,
            device,
            args.batch_size,
        )
        projection_rows, metrics = evaluate_grid(base, truth, path_pdp, device)
        for candidate in projection_rows:
            candidate["amplitude_blend"] = blend
            candidate["compound_name"] = (
                f"blend{blend:.2f}_{candidate['candidate']}"
            )
        grids.extend(projection_rows)
        all_metrics[f"blend_{blend:.2f}"] = metrics
        print(f"path PDP predictor blend {blend:.2f} evaluated", flush=True)
    grids.sort(key=lambda row: float(row["score"]), reverse=True)
    candidates = [
        row
        for row in grids
        if str(row["candidate"]).startswith("path_pdp_")
        and float(row["amplitude_blend"]) > 0.0
    ]
    selected = candidates[0]
    selected_metrics = all_metrics[f"blend_{float(selected['amplitude_blend']):.2f}"]
    selected_names = selected_metrics["candidate_names"].astype(str)
    selected_index = int(np.flatnonzero(selected_names == selected["candidate"])[0])
    baseline_index = int(np.flatnonzero(selected_names == "v9_identity")[0])
    fold_deltas = []
    for fold in range(5):
        local = fold_id == fold
        current = statistics(
            selected_metrics["pas"][selected_index],
            selected_metrics["pdp"][selected_index],
            selected_metrics["error"][selected_index],
            selected_metrics["energy"][selected_index],
            local,
        )
        baseline = statistics(
            selected_metrics["pas"][baseline_index],
            selected_metrics["pdp"][baseline_index],
            selected_metrics["error"][baseline_index],
            selected_metrics["energy"][baseline_index],
            local,
        )
        fold_deltas.append(
            {
                "fold": fold,
                "result": current,
                "baseline": baseline,
                "delta": float(current["score"] - baseline["score"]),
            }
        )
    truth_tensor = torch.as_tensor(truth, device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    base_values = point_metrics(
        torch.as_tensor(base, device=device), truth_tensor, truth_pas, truth_pdp
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Relative path-energy corrections are predictable across spatial holes from path "
            "delay, angle, interaction, stability, geometry, and array-coefficient features."
        ),
        "configuration": vars(args),
        "point_count": int(len(rows)),
        "feature_shape": list(features.shape),
        "fold_diagnostics": fold_diagnostics,
        "v9_baseline": statistics(*base_values),
        "selected": selected,
        "selected_fold_deltas_vs_v9": fold_deltas,
        "positive_folds": int(sum(row["delta"] > 0.0 for row in fold_deltas)),
        "top_candidates": grids[:20],
        "artifacts": {
            "features": PATHS.relative(feature_path),
            "prediction": PATHS.relative(
                ARTIFACT_DIR / "oof_log_amplitude_prediction.npy"
            ),
            "models": model_paths,
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
