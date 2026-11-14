#!/usr/bin/env python3
"""Calibrate multi-seed Sionna PAS statistics against official train channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import generate_v549_multiseed_v207_pas_statistics as source
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 549
VERSION = "v549_multiseed_pas_calibration"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER) / "pas_calibration"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OFFICIAL = PATHS.round1 / "Round1_Train_Channel.npy"
BASE = PATHS.cache_dir(540) / "v540_Round1_Test_Channel.npy"
EXTERNAL_MEMBER = "Round1_Test_Channel.npy"
CHECKPOINTS = (6, 12, 24)
QUALITY_SCALES = (0.0, 1.0, 2.0, 4.0, 8.0)
ALPHAS = (0.125, 0.25, 0.375, 0.50, 0.75, 1.0)
REGIMES = ("clear", "light", "medium", "heavy")
MARGINAL_WEIGHTS = {"sh": 0.4, "psh": 0.2, "shue": 0.2, "shv": 0.2}
FLOOR = 1e-12
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-archive",
        type=Path,
        help="optional external output used only for a target-similarity diagnostic",
    )
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="analyze only checkpoints that currently exist",
    )
    return parser.parse_args()


def checkpoint_path(count: int, split: str, name: str, kind: str) -> Path:
    return source.OUT_DIR / "checkpoints" / f"n{count:02d}_{split}_{name}_{kind}.npy"


def truth_path(name: str) -> Path:
    return OUT_DIR / "official_train" / f"official_train_{name}.npy"


def normalize_rows(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    output /= np.maximum(np.sum(output, axis=1, keepdims=True, dtype=np.float64), TINY)
    return output.astype(np.float32)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, axis=1, dtype=np.float64)
        * np.sum(right * right, axis=1, dtype=np.float64)
    )
    return (numerator / np.maximum(denominator, TINY)).astype(np.float32)


def flat(value: np.ndarray) -> np.ndarray:
    return value.reshape(len(value), -1)


def build_official_truth() -> dict[str, Path]:
    expected_rows = 2000
    paths = {name: truth_path(name) for name in source.MARGINALS}
    if all(path.is_file() for path in paths.values()):
        return paths
    paths["sh"].parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=source.marginal_shape("train", name, expected_rows),
        )
        for name, path in paths.items()
    }
    channel = np.load(OFFICIAL, mmap_mode="r")
    if channel.shape != (expected_rows, 256, 4, 192):
        raise ValueError(f"unexpected official train shape {channel.shape}")
    for start in range(0, expected_rows, 16):
        end = min(start + 16, expected_rows)
        values = source.pas_marginals(
            np.asarray(channel[start:end], dtype=np.complex64)
        )
        for name in source.MARGINALS:
            outputs[name][start:end] = values[name]
        if end == expected_rows or end % 160 == 0:
            print(f"v549 official PAS truth {end}/{expected_rows}", flush=True)
    for output in outputs.values():
        output.flush()
    return paths


def neighbor_plan(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    k: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    xy = positions[:, :2].astype(np.float32)
    distance = np.linalg.norm(
        xy[query_ids, None, :] - xy[None, fit_ids, :], axis=2
    ).astype(np.float32)
    blocked_delta = np.abs(blocked[query_ids, None] - blocked[None, fit_ids]).astype(
        np.float32
    )
    same = regimes[query_ids, None] == regimes[None, fit_ids]
    effective = distance + 0.09 * blocked_delta + 5.0 * (~same)
    local_k = min(k, len(fit_ids))
    nearest = np.argpartition(effective, local_k - 1, axis=1)[:, :local_k]
    rows = np.arange(len(query_ids))[:, None]
    nearest = nearest[rows, np.argsort(effective[rows, nearest], axis=1)]
    selected = effective[rows, nearest]
    selected_same = same[rows, nearest].astype(np.float32)
    weights = np.exp(-selected / 20.0).astype(np.float32) * (
        0.45 + 0.55 * selected_same
    )
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    return fit_ids[nearest], weights


def apply_neighbor_plan(
    truth: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    output = np.empty((len(indices), truth.shape[1]), dtype=np.float32)
    for start in range(0, len(indices), 8):
        end = min(start + 8, len(indices))
        output[start:end] = np.sum(
            np.asarray(truth[indices[start:end]], dtype=np.float32)
            * weights[start:end, :, None],
            axis=1,
        )
    return normalize_rows(output)


def fit_ratio(
    truth: np.ndarray,
    source_values: np.ndarray,
    fit_ids: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, np.ndarray]:
    dimension = truth.shape[1]
    output = {
        name: np.empty(dimension, dtype=np.float32) for name in ("global", *REGIMES)
    }
    for start in range(0, dimension, 1024):
        end = min(start + 1024, dimension)
        delta = np.clip(
            np.log(np.maximum(truth[fit_ids, start:end], FLOOR))
            - np.log(np.maximum(source_values[fit_ids, start:end], FLOOR)),
            -3.0,
            3.0,
        )
        global_ratio = np.median(delta, axis=0).astype(np.float32)
        output["global"][start:end] = global_ratio
        for regime in REGIMES:
            local = regimes[fit_ids] == regime
            if np.sum(local) < 40:
                output[regime][start:end] = global_ratio
                continue
            local_ratio = np.median(delta[local], axis=0).astype(np.float32)
            count = int(np.sum(local))
            shrink = np.float32(count / (count + 120.0))
            output[regime][start:end] = (
                shrink * local_ratio + (1.0 - shrink) * global_ratio
            )
    return output


def apply_ratio(
    source_values: np.ndarray,
    query_ids: np.ndarray,
    regimes: np.ndarray,
    ratios: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.empty((len(query_ids), source_values.shape[1]), dtype=np.float32)
    for regime in REGIMES:
        local = regimes[query_ids] == regime
        if not np.any(local):
            continue
        output[local] = np.asarray(source_values[query_ids[local]], dtype=np.float32)
        output[local] *= np.exp(ratios[regime])[None, :]
    return normalize_rows(output)


def point_instability(count: int, split: str) -> np.ndarray:
    rows = 2000 if split == "train" else 500
    components = []
    for name in source.MARGINALS:
        variance = np.load(
            checkpoint_path(count, split, name, "variance"), mmap_mode="r"
        )
        values = np.empty(rows, dtype=np.float32)
        for start in range(0, rows, 32):
            end = min(start + 32, rows)
            values[start:end] = np.sqrt(
                np.sum(
                    np.asarray(variance[start:end], dtype=np.float32).reshape(
                        end - start, -1
                    ),
                    axis=1,
                    dtype=np.float64,
                )
            ).astype(np.float32)
        components.append(values)
    return np.mean(np.stack(components, axis=1), axis=1).astype(np.float32)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def available_checkpoints(available_only: bool) -> list[int]:
    available = [
        count
        for count in CHECKPOINTS
        if all(
            checkpoint_path(count, split, name, kind).is_file()
            for split in ("train", "test")
            for name in source.MARGINALS
            for kind in ("mean", "variance")
        )
    ]
    if not available_only and available != list(CHECKPOINTS):
        missing = sorted(set(CHECKPOINTS) - set(available))
        raise FileNotFoundError(f"missing v549 checkpoints: {missing}")
    if not available:
        raise FileNotFoundError("no complete v549 checkpoint is available")
    return available


def analyze_checkpoint(
    count: int,
    truth_paths: dict[str, Path],
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    rows = len(positions)
    all_ids = np.arange(rows, dtype=np.int64)
    fold_id = np.full(rows, -1, dtype=np.int16)
    score_bank: dict[str, dict[str, np.ndarray]] = {}
    oof: dict[str, dict[str, np.memmap]] = {}
    checkpoint_dir = OUT_DIR / f"n{count:02d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in source.MARGINALS:
        dimension = int(np.prod(source.marginal_shape("train", name, rows)[1:]))
        oof[name] = {
            kind: np.lib.format.open_memmap(
                checkpoint_dir / f"{name}_{kind}_oof.npy",
                mode="w+",
                dtype=np.float32,
                shape=(rows, dimension),
            )
            for kind in ("baseline", "calibrated")
        }

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        fold_id[query_ids] = fold
        indices, weights = neighbor_plan(
            fit_ids, query_ids, positions, blocked, regimes
        )
        for name in source.MARGINALS:
            truth = flat(np.load(truth_paths[name], mmap_mode="r"))
            raw = flat(
                np.load(checkpoint_path(count, "train", name, "mean"), mmap_mode="r")
            )
            oof[name]["baseline"][query_ids] = apply_neighbor_plan(
                truth, indices, weights
            )
            ratios = fit_ratio(truth, raw, fit_ids, regimes)
            oof[name]["calibrated"][query_ids] = apply_ratio(
                raw, query_ids, regimes, ratios
            )
        print(f"v549 checkpoint={count} calibrated fold={fold + 1}/5", flush=True)
    for values in oof.values():
        for value in values.values():
            value.flush()

    instability = point_instability(count, "train")
    metric_rows: list[dict[str, object]] = []
    candidate_scores: dict[tuple[float, float], np.ndarray] = {}
    per_marginal = {}
    for name in source.MARGINALS:
        truth = flat(np.load(truth_paths[name], mmap_mode="r"))
        raw = flat(
            np.load(checkpoint_path(count, "train", name, "mean"), mmap_mode="r")
        )
        baseline = np.asarray(oof[name]["baseline"], dtype=np.float32)
        calibrated = np.asarray(oof[name]["calibrated"], dtype=np.float32)
        per_marginal[name] = {
            "baseline": row_cosine(truth, baseline),
            "raw": row_cosine(truth, raw),
            "calibrated": row_cosine(truth, calibrated),
        }

    for quality_scale in QUALITY_SCALES:
        quality = np.exp(-quality_scale * instability).astype(np.float32)
        for alpha in ALPHAS:
            effective = np.clip(alpha * quality, 0.0, 1.0).astype(np.float32)
            composite = np.zeros(rows, dtype=np.float32)
            marginal_means = {}
            for name in source.MARGINALS:
                truth = flat(np.load(truth_paths[name], mmap_mode="r"))
                baseline = np.asarray(oof[name]["baseline"], dtype=np.float32)
                calibrated = np.asarray(oof[name]["calibrated"], dtype=np.float32)
                candidate = normalize_rows(
                    baseline * (1.0 - effective[:, None])
                    + calibrated * effective[:, None]
                )
                scores = row_cosine(truth, candidate)
                composite += np.float32(MARGINAL_WEIGHTS[name]) * scores
                marginal_means[name] = float(np.mean(scores))
            candidate_scores[(quality_scale, alpha)] = composite
            metric_rows.append(
                {
                    "quality_scale": quality_scale,
                    "alpha": alpha,
                    "effective": summarize(effective),
                    "composite": float(np.mean(composite)),
                    "marginals": marginal_means,
                }
            )

    best = max(metric_rows, key=lambda row: row["composite"])
    selected_key = (float(best["quality_scale"]), float(best["alpha"]))
    selected_scores = candidate_scores[selected_key]
    baseline_composite = np.zeros(rows, dtype=np.float32)
    raw_composite = np.zeros(rows, dtype=np.float32)
    calibrated_composite = np.zeros(rows, dtype=np.float32)
    for name in source.MARGINALS:
        weight = np.float32(MARGINAL_WEIGHTS[name])
        baseline_composite += weight * per_marginal[name]["baseline"]
        raw_composite += weight * per_marginal[name]["raw"]
        calibrated_composite += weight * per_marginal[name]["calibrated"]

    shuffled = np.empty(rows, dtype=np.int64)
    rng = np.random.default_rng(549000 + count)
    for fold in range(5):
        for regime in REGIMES:
            ids = np.flatnonzero((fold_id == fold) & (regimes == regime))
            shuffled[ids] = rng.permutation(ids)
    quality = np.exp(-selected_key[0] * instability).astype(np.float32)
    effective = np.clip(selected_key[1] * quality, 0.0, 1.0).astype(np.float32)
    shuffled_composite = np.zeros(rows, dtype=np.float32)
    for name in source.MARGINALS:
        truth = flat(np.load(truth_paths[name], mmap_mode="r"))
        baseline = np.asarray(oof[name]["baseline"], dtype=np.float32)
        calibrated = np.asarray(oof[name]["calibrated"], dtype=np.float32)
        candidate = normalize_rows(
            baseline * (1.0 - effective[:, None])
            + calibrated[shuffled] * effective[:, None]
        )
        shuffled_composite += np.float32(MARGINAL_WEIGHTS[name]) * row_cosine(
            truth, candidate
        )

    fold_rows = []
    for fold in range(5):
        mask = fold_id == fold
        fold_rows.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "baseline": float(np.mean(baseline_composite[mask])),
                "selected": float(np.mean(selected_scores[mask])),
                "gain": float(
                    np.mean(selected_scores[mask] - baseline_composite[mask])
                ),
            }
        )
    result = {
        "checkpoint": count,
        "instability": summarize(instability),
        "baseline": summarize(baseline_composite),
        "raw_sionna": summarize(raw_composite),
        "calibrated_sionna": summarize(calibrated_composite),
        "selected": {
            **best,
            "scores": summarize(selected_scores),
            "gain": float(np.mean(selected_scores - baseline_composite)),
            "fold_wins": int(sum(row["gain"] > 0.0 for row in fold_rows)),
            "folds": fold_rows,
            "by_regime": {
                regime: {
                    "count": int(np.sum(regimes == regime)),
                    "baseline": float(np.mean(baseline_composite[regimes == regime])),
                    "selected": float(np.mean(selected_scores[regimes == regime])),
                    "gain": float(
                        np.mean(
                            selected_scores[regimes == regime]
                            - baseline_composite[regimes == regime]
                        )
                    ),
                }
                for regime in REGIMES
            },
        },
        "shuffled_control": {
            "score": float(np.mean(shuffled_composite)),
            "gain_vs_baseline": float(np.mean(shuffled_composite - baseline_composite)),
            "real_minus_shuffled": float(np.mean(selected_scores - shuffled_composite)),
        },
        "grid": metric_rows,
        "marginals": {
            name: {key: summarize(values) for key, values in metrics.items()}
            for name, metrics in per_marginal.items()
        },
    }
    arrays = {
        "fold_id": fold_id,
        "instability": instability,
        "baseline_composite": baseline_composite,
        "selected_composite": selected_scores,
    }
    return result, arrays


def build_test_targets(
    count: int,
    selected: dict[str, object],
    truth_paths: dict[str, Path],
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> tuple[dict[str, Path], dict[str, dict[str, Path]]]:
    train_ids = np.arange(len(train_positions), dtype=np.int64)
    test_ids = np.arange(len(test_positions), dtype=np.int64)
    combined_positions = np.concatenate((train_positions, test_positions), axis=0)
    combined_blocked = np.concatenate((train_blocked, test_blocked), axis=0)
    combined_regimes = np.concatenate((train_regimes, test_regimes), axis=0)
    combined_fit = train_ids
    combined_query = len(train_positions) + test_ids
    indices, weights = neighbor_plan(
        combined_fit,
        combined_query,
        combined_positions,
        combined_blocked,
        combined_regimes,
    )
    instability = point_instability(count, "test")
    quality_scale = float(selected["quality_scale"])
    alpha = float(selected["alpha"])
    quality = np.exp(-quality_scale * instability).astype(np.float32)
    effective = np.clip(alpha * quality, 0.0, 1.0).astype(np.float32)
    target_dir = OUT_DIR / "test_targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    components: dict[str, dict[str, Path]] = {}
    for name in source.MARGINALS:
        train_truth = flat(np.load(truth_paths[name], mmap_mode="r"))
        train_source = flat(
            np.load(checkpoint_path(count, "train", name, "mean"), mmap_mode="r")
        )
        test_source = flat(
            np.load(checkpoint_path(count, "test", name, "mean"), mmap_mode="r")
        )
        baseline = apply_neighbor_plan(train_truth, indices, weights)
        ratios = fit_ratio(train_truth, train_source, train_ids, train_regimes)
        calibrated = np.empty_like(test_source, dtype=np.float32)
        for regime in REGIMES:
            local = test_regimes == regime
            if not np.any(local):
                continue
            calibrated[local] = np.asarray(test_source[local], dtype=np.float32)
            calibrated[local] *= np.exp(ratios[regime])[None, :]
        calibrated = normalize_rows(calibrated)
        target = normalize_rows(
            baseline * (1.0 - effective[:, None]) + calibrated * effective[:, None]
        )
        log_action = np.clip(
            np.log(np.maximum(target, FLOOR)) - np.log(np.maximum(baseline, FLOOR)),
            -3.0,
            3.0,
        ).astype(np.float32)
        path = target_dir / f"n{count:02d}_test_{name}_target.npy"
        np.save(path, target.reshape(source.marginal_shape("test", name, 500)))
        baseline_path = target_dir / f"n{count:02d}_test_{name}_baseline.npy"
        calibrated_path = target_dir / f"n{count:02d}_test_{name}_calibrated.npy"
        action_path = target_dir / f"n{count:02d}_test_{name}_log_action.npy"
        np.save(
            baseline_path,
            baseline.reshape(source.marginal_shape("test", name, 500)),
        )
        np.save(
            calibrated_path,
            calibrated.reshape(source.marginal_shape("test", name, 500)),
        )
        np.save(
            action_path,
            log_action.reshape(source.marginal_shape("test", name, 500)),
        )
        outputs[name] = path
        components[name] = {
            "baseline": baseline_path,
            "calibrated": calibrated_path,
            "log_action": action_path,
        }
    np.savez_compressed(
        target_dir / f"n{count:02d}_test_target_quality.npz",
        instability=instability,
        quality=quality,
        effective=effective,
        regimes=test_regimes,
    )
    return outputs, components


def channel_marginals(path: Path, rows: int) -> dict[str, np.ndarray]:
    channel = np.load(path, mmap_mode="r")
    outputs = {
        name: np.empty(source.marginal_shape("test", name, rows), dtype=np.float32)
        for name in source.MARGINALS
    }
    for start in range(0, rows, 2):
        end = min(start + 2, rows)
        values = source.pas_marginals(
            np.asarray(channel[start:end], dtype=np.complex64)
        )
        for name in source.MARGINALS:
            outputs[name][start:end] = values[name]
    return outputs


def external_marginals(archive_path: Path, rows: int) -> dict[str, np.ndarray]:
    outputs = {
        name: np.empty(source.marginal_shape("test", name, rows), dtype=np.float32)
        for name in source.MARGINALS
    }
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("external archive CRC failed")
        with archive.open(EXTERNAL_MEMBER) as handle:
            version = np.lib.format.read_magic(handle)
            shape, fortran_order, dtype = np.lib.format._read_array_header(
                handle, version
            )
            if tuple(shape) != (rows, 256, 4, 192) or fortran_order:
                raise ValueError(f"unexpected external NPY header: {shape}")
            row_values = int(np.prod(shape[1:]))
            for row in range(rows):
                raw = handle.read(row_values * dtype.itemsize)
                channel = np.frombuffer(raw, dtype=dtype).reshape(1, *shape[1:])
                values = source.pas_marginals(channel)
                for name in source.MARGINALS:
                    outputs[name][row] = values[name][0]
    return outputs


def target_similarity(
    targets: dict[str, Path], external_archive: Path | None
) -> dict[str, object]:
    base = channel_marginals(BASE, 500)
    comparisons = {}
    teachers = {"v540": base}
    if external_archive is not None:
        teachers["external"] = external_marginals(external_archive, 500)
    for teacher_name, teacher in teachers.items():
        marginal_scores = {}
        composite = np.zeros(500, dtype=np.float32)
        for name in source.MARGINALS:
            target = flat(np.load(targets[name], mmap_mode="r"))
            scores = row_cosine(target, flat(teacher[name]))
            marginal_scores[name] = summarize(scores)
            composite += np.float32(MARGINAL_WEIGHTS[name]) * scores
        comparisons[teacher_name] = {
            "composite": summarize(composite),
            "marginals": marginal_scores,
        }
    return comparisons


def main() -> None:
    args = parse_args()
    for path in (OFFICIAL, BASE):
        if not path.is_file():
            raise FileNotFoundError(path)
    external_archive = (
        args.external_archive.resolve() if args.external_archive else None
    )
    if external_archive is not None and not external_archive.is_file():
        raise FileNotFoundError(external_archive)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    counts = available_checkpoints(args.available_only)
    truth_paths = build_official_truth()
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geometry["train_blocked"].astype(np.float32)
    test_blocked = geometry["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked).astype("U8")
    test_regimes = regime_of(test_blocked).astype("U8")
    all_ids = np.arange(len(train_positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, train_positions[:, :2])

    results = []
    arrays = {}
    for count in counts:
        result, local_arrays = analyze_checkpoint(
            count,
            truth_paths,
            train_positions,
            train_blocked,
            train_regimes,
            folds,
        )
        results.append(result)
        arrays.update(
            {f"n{count:02d}_{key}": value for key, value in local_arrays.items()}
        )
    best = max(results, key=lambda item: item["selected"]["composite"])
    selected_count = int(best["checkpoint"])
    test_targets, test_components = build_test_targets(
        selected_count,
        best["selected"],
        truth_paths,
        train_positions,
        test_positions,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
    )
    similarity = target_similarity(test_targets, external_archive)
    arrays_path = OUT_DIR / "v549_multiseed_pas_calibration_oof.npz"
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A Monte Carlo PAS expectation and its variance provide more stable same-coordinate "
            "path information than a single Sionna realization; official-train log-ratio "
            "calibration removes shared simulation bias."
        ),
        "evidence_boundary": {
            "selection": "five mutually exclusive official-train spatial folds",
            "online_score_used": False,
            "external_output_role": (
                "target-similarity diagnostic only" if external_archive else "not used"
            ),
        },
        "marginal_weights": MARGINAL_WEIGHTS,
        "checkpoints": results,
        "selected": {
            "checkpoint": selected_count,
            **best["selected"],
            "test_target_similarity": similarity,
        },
        "outputs": {
            "oof": PATHS.relative(arrays_path),
            "test_targets": {
                name: PATHS.relative(path) for name, path in test_targets.items()
            },
            "test_components": {
                name: {key: PATHS.relative(path) for key, path in paths.items()}
                for name, paths in test_components.items()
            },
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
