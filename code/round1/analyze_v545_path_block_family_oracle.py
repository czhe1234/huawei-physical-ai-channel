#!/usr/bin/env python3
"""Audit whether the v543 cell oracle survives path and spatial constraints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.spatial import cKDTree

import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v494_sionna_configuration_stacking as v494
import analyze_v498_sionna_path_lifecycle_profiles as v498
import analyze_v531_expanded_sbi_configuration_library as v531
import analyze_v541_corrected_full_library_sbi as v541
import generate_v543_disagreement_conditioned_cell_fusion as v543
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 545
VERSION = "v545_path_block_family_oracle"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT_DIR = PATHS.analysis / "visualizations" / "训练集路径块与路径族上限"

OOF_SOURCE = PATHS.cache_dir(543) / "v543_cellwise_fusion_oof.npz"
BANK_CACHE = CACHE_DIR / "v545_oof_candidate_bank.npy"
BANK_META = CACHE_DIR / "v545_oof_candidate_bank.json"
PATH_META = CACHE_DIR / "v545_path_family_geometry.npz"
SCORE_CACHE = CACHE_DIR / "v545_path_family_oracle_scores.npz"
POINT_ORACLE_CACHE = CACHE_DIR / "v545_point_path_block_oracle.npy"
CROSSFIT_CACHE = CACHE_DIR / "v545_best_crossfit_path_family.npy"
PATH_ROOT = PATHS.cache_dir(475) / "shards"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TRAIN_POWER = PATHS.shared_features / "train_total_power.npy"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
SCENE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"

FIGURES = {
    "ladder": OUTPUT_DIR / "01_物理约束逐级增加后的上限曲线.png",
    "grid": OUTPUT_DIR / "02_连续谱块大小敏感性.png",
    "family": OUTPUT_DIR / "03_路径族身份与空间尺度敏感性.png",
    "space": OUTPUT_DIR / "04_重遮挡点路径族改进空间分布.png",
    "examples": OUTPUT_DIR / "05_典型重遮挡点从逐格碎片到路径族约束.png",
    "coverage": OUTPUT_DIR / "06_路径族复现频率与交叉拟合覆盖率.png",
}
SUMMARY_CSV = OUTPUT_DIR / "路径块与路径族上限汇总.csv"

REGIMES = ("clear", "light", "medium", "heavy")
REGIME_CN = {
    "clear": "无遮挡",
    "light": "轻遮挡",
    "medium": "中遮挡",
    "heavy": "重遮挡",
}
IDENTITY_MODES = ("exact", "endpoint", "first")
IDENTITY_CN = {
    "exact": "完整交互物体序列",
    "endpoint": "首尾物体与深度",
    "first": "首个物体与机制",
}
HEIGHT = 192
WIDTH = 16
SOURCE_COUNT = 19
MAX_PATHS = 128
OWNER_NONE = np.uint8(255)
GAIN_MIN = 0.25
GAIN_MAX = 4.0
GRID_BLOCKS = (
    (1, 1),
    (1, 2),
    (3, 2),
    (3, 4),
    (6, 4),
    (12, 8),
    (24, 16),
    (192, 16),
)
LOCAL_RADII = (20.0, 40.0, 80.0, 160.0)
LOCAL_NEIGHBORS = 8
MIN_FAMILY_SUPPORT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-bank", action="store_true")
    parser.add_argument("--rebuild-path-meta", action="store_true")
    return parser.parse_args()


def configure_chinese_font() -> None:
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
            font_manager.fontManager.addfont(path)
            selected = font_manager.FontProperties(fname=path).get_name()
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 260,
        }
    )


def normalize_rows(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return output / np.maximum(output.sum(axis=1, keepdims=True), TINY)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    right = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    return (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)


def summarize(value: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
    }


def validate_inputs() -> None:
    required = (
        OOF_SOURCE,
        TRAIN_POS,
        TRAIN_POWER,
        GEOMETRY,
        SCENE,
        v434.OOF_CACHE,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    shards = sorted(PATH_ROOT.glob("shard_*/path_targets.npz"))
    if len(shards) != 25:
        missing.append(f"expected 25 v475 path shards, found {len(shards)}")
    if missing:
        raise FileNotFoundError(f"missing v545 inputs: {missing}")


def load_train_sources() -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[str]]:
    truth, old_sources, old_quality, old_names = v531.load_expanded_sources()
    corrected_truth, corrected_sources, corrected_quality, corrected_names = (
        v541.load_corrected_sources()
    )
    if not np.allclose(truth, corrected_truth):
        raise ValueError("old and corrected truth mismatch")
    sources = list(old_sources) + list(corrected_sources)
    names = [f"old_{name}" for name in old_names] + list(corrected_names)
    quality = np.maximum(old_quality, corrected_quality).astype(np.float32)
    if len(sources) != SOURCE_COUNT - 1:
        raise ValueError(f"expected 18 physical sources, found {len(sources)}")
    return truth.astype(np.float32), sources, quality, names


def build_oof_bank(rebuild: bool) -> tuple[np.memmap, list[str], dict[str, object]]:
    if BANK_CACHE.is_file() and BANK_META.is_file() and not rebuild:
        metadata = json.loads(BANK_META.read_text(encoding="utf-8"))
        bank = np.load(BANK_CACHE, mmap_mode="r")
        if bank.shape != (2000, SOURCE_COUNT, HEIGHT * WIDTH):
            raise ValueError(f"invalid bank cache shape: {bank.shape}")
        return bank, metadata["source_names"], metadata

    truth, sources, quality, names = load_train_sources()
    positions = np.load(TRAIN_POS).astype(np.float32)
    geometry = np.load(GEOMETRY)
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    fold_id = np.load(v434.OOF_CACHE)["fold_id"].astype(np.int16)
    stored_anchor = np.load(OOF_SOURCE)["anchor"].astype(np.float32)
    partial = BANK_CACHE.with_name(BANK_CACHE.stem + ".partial.npy")
    if partial.exists():
        partial.unlink()
    bank = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float32,
        shape=(len(truth), SOURCE_COUNT, truth.shape[1]),
    )
    anchor_error: list[float] = []

    for outer in range(5):
        query_ids = np.flatnonzero(fold_id == outer)
        fit_ids = np.flatnonzero(fold_id != outer)
        inner_baseline = np.zeros_like(truth)
        inner_candidates = np.zeros(
            (len(sources), len(truth), truth.shape[1]), dtype=np.float32
        )
        for inner in range(5):
            if inner == outer:
                continue
            inner_query = np.flatnonzero(fold_id == inner)
            inner_fit = np.flatnonzero((fold_id != outer) & (fold_id != inner))
            inner_baseline[inner_query] = normalize_rows(
                v432.neighbor_prediction(
                    inner_fit,
                    inner_query,
                    positions,
                    blocked,
                    regimes,
                    truth,
                    k=48,
                )
            )
            inner_candidates[:, inner_query] = v494.calibrated_sources(
                truth, sources, inner_fit, inner_query, regimes
            )

        query_baseline = normalize_rows(
            v432.neighbor_prediction(
                fit_ids,
                query_ids,
                positions,
                blocked,
                regimes,
                truth,
                k=48,
            )
        )
        query_candidates = v494.calibrated_sources(
            truth, sources, fit_ids, query_ids, regimes
        )
        weights: dict[str, np.ndarray] = {}
        for regime in REGIMES:
            regime_fit = fit_ids[regimes[fit_ids] == regime]
            weights[regime] = v494.fit_convex_weights(
                truth, inner_candidates, regime_fit
            )
        inner_target = np.zeros_like(truth)
        inner_target[fit_ids] = v494.weighted_target(
            inner_candidates, fit_ids, regimes, weights
        )
        query_target = v494.weighted_target(
            query_candidates,
            np.arange(len(query_ids), dtype=np.int64),
            regimes[query_ids],
            weights,
        )
        alpha = v494.select_alpha(
            truth,
            inner_baseline,
            inner_target,
            quality,
            fit_ids,
            regimes,
        )
        query_anchor, _ = v543.blend_local(
            query_baseline,
            query_target,
            quality[query_ids],
            regimes[query_ids],
            alpha,
        )
        anchor_error.append(
            float(np.max(np.abs(query_anchor - stored_anchor[query_ids])))
        )
        bank[query_ids, 0] = query_baseline
        bank[query_ids, 1:] = query_candidates.transpose(1, 0, 2)
        bank.flush()
        print(
            f"v545 OOF bank fold {outer + 1}/5 "
            f"anchor_max_abs={anchor_error[-1]:.3e}",
            flush=True,
        )
        del inner_baseline, inner_candidates, query_candidates, query_baseline
        gc.collect()

    bank.flush()
    del bank
    partial.replace(BANK_CACHE)
    metadata = {
        "shape": [2000, SOURCE_COUNT, HEIGHT * WIDTH],
        "dtype": "float32",
        "source_names": ["official_48nn_baseline", *names],
        "fold_anchor_max_abs": anchor_error,
        "maximum_anchor_abs_error": float(max(anchor_error)),
        "query_truth_used_for_calibration": False,
    }
    BANK_META.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return np.load(BANK_CACHE, mmap_mode="r"), metadata["source_names"], metadata


def family_key(
    interactions: np.ndarray,
    objects: np.ndarray,
    count: int,
    mask: int,
    mode: str,
) -> tuple[int, ...]:
    if count == 0:
        return (0,)
    local_interactions = interactions[:count].astype(np.int64)
    local_objects = objects[:count].astype(np.int64)
    if mode == "exact":
        pairs = np.column_stack([local_interactions, local_objects]).ravel()
        return (count, *[int(value) for value in pairs])
    if mode == "endpoint":
        return (
            count,
            mask,
            int(local_interactions[0]),
            int(local_objects[0]),
            int(local_interactions[-1]),
            int(local_objects[-1]),
        )
    if mode == "first":
        return (
            min(count, 4),
            int(local_interactions[0]),
            int(local_objects[0]),
        )
    raise ValueError(mode)


def build_path_metadata(rebuild: bool) -> dict[str, np.ndarray]:
    if PATH_META.is_file() and not rebuild:
        cached = np.load(PATH_META)
        return {key: np.asarray(cached[key]) for key in cached.files}

    owner_path = np.full((2000, HEIGHT * WIDTH), OWNER_NONE, dtype=np.uint8)
    path_valid = np.zeros((2000, MAX_PATHS), dtype=bool)
    family_arrays = {
        mode: np.full((2000, MAX_PATHS), -1, dtype=np.int32) for mode in IDENTITY_MODES
    }
    family_maps: dict[str, dict[tuple[int, ...], int]] = {
        mode: {} for mode in IDENTITY_MODES
    }
    atom_cosine = np.zeros(2000, dtype=np.float32)
    path_count = np.zeros(2000, dtype=np.int16)
    offset = 0

    for shard_id in range(20):
        shard = PATH_ROOT / f"shard_{shard_id:05d}"
        with np.load(shard / "path_targets.npz") as paths, np.load(
            shard / "spectral_targets.npz"
        ) as spectra:
            rows = len(paths["valid"])
            for row in range(rows):
                point = offset + row
                valid_ids = np.flatnonzero(paths["valid"][row])
                path_count[point] = len(valid_ids)
                if not len(valid_ids):
                    continue
                path_valid[point, valid_ids] = True
                tau = paths["tau"][row, valid_ids].astype(np.float32)
                delay = v498.delay_fingerprint(tau)
                horizontal = paths["tx_h_fingerprint"][row, valid_ids].astype(
                    np.float32
                )
                horizontal /= np.maximum(horizontal.sum(axis=1, keepdims=True), TINY)
                mass = paths["normalized_power"][row, valid_ids].astype(np.float32)
                mass /= max(float(mass.sum()), TINY)
                atoms = mass[:, None, None] * delay[:, :, None] * horizontal[:, None, :]
                owner_local = np.argmax(atoms, axis=0)
                owner_path[point] = valid_ids[owner_local].reshape(-1).astype(np.uint8)
                reconstructed = atoms.sum(axis=0, dtype=np.float64).reshape(1, -1)
                target = spectra["joint_sh"][row].astype(np.float32).reshape(1, -1)
                atom_cosine[point] = row_cosine(reconstructed, target)[0]

                for original_path in valid_ids:
                    count = int(paths["interaction_count"][row, original_path])
                    mask = int(paths["interaction_mask"][row, original_path])
                    for mode in IDENTITY_MODES:
                        key = family_key(
                            paths["interactions"][row, original_path],
                            paths["objects"][row, original_path],
                            count,
                            mask,
                            mode,
                        )
                        mapping = family_maps[mode]
                        if key not in mapping:
                            mapping[key] = len(mapping)
                        family_arrays[mode][point, original_path] = mapping[key]
        offset += rows
        print(
            f"v545 path metadata {offset}/2000 "
            + " ".join(f"{mode}={len(family_maps[mode])}" for mode in IDENTITY_MODES),
            flush=True,
        )
    if offset != 2000:
        raise ValueError(f"expected 2000 train paths, found {offset}")
    np.savez_compressed(
        PATH_META,
        owner_path=owner_path,
        path_valid=path_valid,
        path_count=path_count,
        atom_cosine=atom_cosine,
        **{f"family_{mode}": value for mode, value in family_arrays.items()},
    )
    return {
        "owner_path": owner_path,
        "path_valid": path_valid,
        "path_count": path_count,
        "atom_cosine": atom_cosine,
        **{f"family_{mode}": value for mode, value in family_arrays.items()},
    }


def score_prediction(
    truth: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    regimes: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metrics = v543.spectrum_scores(truth, prediction)
    summary: dict[str, object] = {
        "overall": {
            key: float(np.mean(value[valid])) for key, value in metrics.items()
        },
        "by_regime": {},
    }
    for regime in REGIMES:
        mask = valid & (regimes == regime)
        summary["by_regime"][regime] = {
            key: float(np.mean(value[mask])) for key, value in metrics.items()
        }
    return summary, metrics


def grid_block_oracle_point(
    truth: np.ndarray,
    bank: np.ndarray,
    delay_size: int,
    angle_size: int,
) -> np.ndarray:
    if HEIGHT % delay_size or WIDTH % angle_size:
        raise ValueError("grid block dimensions must divide the spectrum")
    delay_blocks = HEIGHT // delay_size
    angle_blocks = WIDTH // angle_size
    candidates = bank.reshape(
        SOURCE_COUNT, delay_blocks, delay_size, angle_blocks, angle_size
    )
    target = truth.reshape(delay_blocks, delay_size, angle_blocks, angle_size)
    error = np.sum(
        (candidates.astype(np.float64) - target[None].astype(np.float64)) ** 2,
        axis=(2, 4),
    )
    chosen = np.argmin(error, axis=0)
    output = np.empty_like(target, dtype=np.float32)
    for delay_block in range(delay_blocks):
        for angle_block in range(angle_blocks):
            output[delay_block, :, angle_block, :] = candidates[
                chosen[delay_block, angle_block],
                delay_block,
                :,
                angle_block,
                :,
            ]
    output = output.reshape(-1)
    output /= max(float(output.sum()), TINY)
    return output


def compute_grid_curve(
    truth: np.ndarray,
    bank: np.ndarray,
    valid: np.ndarray,
    regimes: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    summaries: dict[str, object] = {}
    point_scores: dict[str, np.ndarray] = {}
    for delay_size, angle_size in GRID_BLOCKS:
        prediction = np.empty_like(truth)
        for point in range(len(truth)):
            prediction[point] = grid_block_oracle_point(
                truth[point], bank[point], delay_size, angle_size
            )
        key = f"d{delay_size}_a{angle_size}"
        summary, metrics = score_prediction(truth, prediction, valid, regimes)
        summaries[key] = summary
        point_scores[key] = metrics["weighted"]
        print(
            f"v545 grid {key} weighted={summary['overall']['weighted']:.6f}",
            flush=True,
        )
        del prediction
    return summaries, point_scores


def owner_family_grid(path_meta: dict[str, np.ndarray], mode: str) -> np.ndarray:
    owner = path_meta["owner_path"]
    family = path_meta[f"family_{mode}"]
    rows = np.arange(len(owner), dtype=np.int64)[:, None]
    safe_owner = np.minimum(owner.astype(np.int64), MAX_PATHS - 1)
    output = family[rows, safe_owner]
    output[owner == OWNER_NONE] = -1
    return output.astype(np.int32)


def build_records(
    truth: np.ndarray,
    bank: np.ndarray,
    owner_family: np.ndarray,
    valid_points: np.ndarray,
) -> dict[str, np.ndarray]:
    points: list[int] = []
    families: list[int] = []
    cells: list[int] = []
    cross: list[np.ndarray] = []
    candidate_sq: list[np.ndarray] = []
    truth_sq: list[float] = []
    for point in np.flatnonzero(valid_points):
        point_family = owner_family[point]
        for family in np.unique(point_family[point_family >= 0]):
            mask = point_family == family
            # Index the point first so NumPy keeps the source axis before the
            # boolean-selected cell axis.
            candidate = np.asarray(bank[point], dtype=np.float64)[:, mask]
            target = np.asarray(truth[point, mask], dtype=np.float64)
            if candidate.shape != (SOURCE_COUNT, target.size):
                raise ValueError(
                    f"invalid record shape at point={point}, family={family}: "
                    f"candidate={candidate.shape}, target={target.shape}"
                )
            points.append(int(point))
            families.append(int(family))
            cells.append(int(np.sum(mask)))
            cross.append(np.sum(candidate * target[None, :], axis=1))
            candidate_sq.append(np.sum(candidate * candidate, axis=1))
            truth_sq.append(float(np.sum(target * target)))
    records = {
        "point": np.asarray(points, dtype=np.int32),
        "family": np.asarray(families, dtype=np.int32),
        "cells": np.asarray(cells, dtype=np.int32),
        "cross": np.asarray(cross, dtype=np.float64),
        "candidate_sq": np.asarray(candidate_sq, dtype=np.float64),
        "truth_sq": np.asarray(truth_sq, dtype=np.float64),
    }
    print(
        f"v545 records={len(points)} families={len(np.unique(records['family']))}",
        flush=True,
    )
    return records


def source_choice(
    cross: np.ndarray,
    candidate_sq: np.ndarray,
    truth_sq: float,
    scaled: bool,
) -> tuple[int, float]:
    if scaled:
        gain = np.clip(cross / np.maximum(candidate_sq, 1e-30), GAIN_MIN, GAIN_MAX)
    else:
        gain = np.ones_like(cross)
    error = truth_sq - 2.0 * gain * cross + gain * gain * candidate_sq
    source = int(np.argmin(error))
    return source, float(gain[source])


def apply_record_choices(
    anchor: np.ndarray,
    bank: np.ndarray,
    owner_family: np.ndarray,
    records: dict[str, np.ndarray],
    choices: np.ndarray,
    gains: np.ndarray,
) -> np.ndarray:
    output = np.asarray(anchor, dtype=np.float32).copy()
    for record in np.flatnonzero(choices >= 0):
        point = int(records["point"][record])
        family = int(records["family"][record])
        source = int(choices[record])
        mask = owner_family[point] == family
        output[point, mask] = np.asarray(
            bank[point, source, mask], dtype=np.float32
        ) * np.float32(gains[record])
    return normalize_rows(output)


def point_oracle_choices(
    records: dict[str, np.ndarray], scaled: bool
) -> tuple[np.ndarray, np.ndarray]:
    choices = np.full(len(records["point"]), -1, dtype=np.int16)
    gains = np.ones(len(choices), dtype=np.float32)
    for record in range(len(choices)):
        choices[record], gains[record] = source_choice(
            records["cross"][record],
            records["candidate_sq"][record],
            float(records["truth_sq"][record]),
            scaled,
        )
    return choices, gains


def grouped_oracle_choices(
    records: dict[str, np.ndarray],
    scaled: bool,
    minimum_occurrences: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    choices = np.full(len(records["point"]), -1, dtype=np.int16)
    gains = np.ones(len(choices), dtype=np.float32)
    used = 0
    for family in np.unique(records["family"]):
        ids = np.flatnonzero(records["family"] == family)
        if len(ids) < minimum_occurrences:
            continue
        source, gain = source_choice(
            np.sum(records["cross"][ids], axis=0),
            np.sum(records["candidate_sq"][ids], axis=0),
            float(np.sum(records["truth_sq"][ids])),
            scaled,
        )
        choices[ids] = source
        gains[ids] = gain
        used += len(ids)
    coverage = {
        "record_fraction": float(used / max(len(choices), 1)),
        "cell_fraction": float(
            np.sum(records["cells"][choices >= 0]) / max(np.sum(records["cells"]), 1)
        ),
    }
    return choices, gains, coverage


def crossfit_family_choices(
    records: dict[str, np.ndarray],
    positions: np.ndarray,
    fold_id: np.ndarray,
    *,
    radius: float | None,
    neighbors: int,
    minimum_support: int,
    scaled: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    choices = np.full(len(records["point"]), -1, dtype=np.int16)
    gains = np.ones(len(choices), dtype=np.float32)
    family_order = np.argsort(records["family"], kind="stable")
    sorted_family = records["family"][family_order]
    boundaries = np.flatnonzero(np.diff(sorted_family)) + 1
    groups = np.split(family_order, boundaries)
    for group in groups:
        points = records["point"][group]
        occurrence_folds = fold_id[points]
        for outer in range(5):
            query = group[occurrence_folds == outer]
            fit = group[occurrence_folds != outer]
            if not len(query) or len(fit) < minimum_support:
                continue
            if radius is None:
                source, gain = source_choice(
                    np.sum(records["cross"][fit], axis=0),
                    np.sum(records["candidate_sq"][fit], axis=0),
                    float(np.sum(records["truth_sq"][fit])),
                    scaled,
                )
                choices[query] = source
                gains[query] = gain
                continue
            tree = cKDTree(positions[records["point"][fit], :2])
            k = min(neighbors, len(fit))
            distance, local_index = tree.query(
                positions[records["point"][query], :2],
                k=k,
                distance_upper_bound=radius,
            )
            if k == 1:
                distance = distance[:, None]
                local_index = local_index[:, None]
            for local_query, record in enumerate(query):
                valid_neighbor = np.isfinite(distance[local_query]) & (
                    local_index[local_query] < len(fit)
                )
                if int(np.sum(valid_neighbor)) < minimum_support:
                    continue
                neighbor_records = fit[local_index[local_query, valid_neighbor]]
                source, gain = source_choice(
                    np.sum(records["cross"][neighbor_records], axis=0),
                    np.sum(records["candidate_sq"][neighbor_records], axis=0),
                    float(np.sum(records["truth_sq"][neighbor_records])),
                    scaled,
                )
                choices[record] = source
                gains[record] = gain
    used = choices >= 0
    coverage = {
        "record_fraction": float(np.mean(used)),
        "cell_fraction": float(
            np.sum(records["cells"][used]) / max(np.sum(records["cells"]), 1)
        ),
        "modified_points": int(len(np.unique(records["point"][used]))),
    }
    return choices, gains, coverage


def method_row(
    name: str,
    summary: dict[str, object],
    anchor_summary: dict[str, object],
    cell_summary: dict[str, object],
    coverage: dict[str, float] | None = None,
) -> dict[str, object]:
    score = float(summary["overall"]["weighted"])
    anchor = float(anchor_summary["overall"]["weighted"])
    cell = float(cell_summary["overall"]["weighted"])
    row: dict[str, object] = {
        "name": name,
        "overall": summary,
        "gain_vs_anchor": score - anchor,
        "cell_oracle_gain_retention": (score - anchor) / max(cell - anchor, 1e-12),
    }
    if coverage is not None:
        row["coverage"] = coverage
    return row


def evaluate_family_mode(
    mode: str,
    truth: np.ndarray,
    anchor: np.ndarray,
    bank: np.ndarray,
    path_meta: dict[str, np.ndarray],
    valid: np.ndarray,
    regimes: np.ndarray,
    positions: np.ndarray,
    fold_id: np.ndarray,
    anchor_summary: dict[str, object],
    cell_summary: dict[str, object],
) -> tuple[
    list[dict[str, object]],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    owner = owner_family_grid(path_meta, mode)
    records = build_records(truth, bank, owner, valid)
    rows: list[dict[str, object]] = []
    scores: dict[str, np.ndarray] = {}

    point_choices, point_gains = point_oracle_choices(records, scaled=False)
    point_prediction = apply_record_choices(
        anchor, bank, owner, records, point_choices, point_gains
    )
    point_summary, point_metrics = score_prediction(
        truth, point_prediction, valid, regimes
    )
    key = f"{mode}_point_source"
    rows.append(method_row(key, point_summary, anchor_summary, cell_summary))
    scores[key] = point_metrics["weighted"]

    scaled_choices, scaled_gains = point_oracle_choices(records, scaled=True)
    scaled_prediction = apply_record_choices(
        anchor, bank, owner, records, scaled_choices, scaled_gains
    )
    scaled_summary, scaled_metrics = score_prediction(
        truth, scaled_prediction, valid, regimes
    )
    key = f"{mode}_point_source_gain"
    rows.append(method_row(key, scaled_summary, anchor_summary, cell_summary))
    scores[key] = scaled_metrics["weighted"]

    global_choices, global_gains, global_coverage = grouped_oracle_choices(
        records, scaled=True, minimum_occurrences=MIN_FAMILY_SUPPORT
    )
    global_prediction = apply_record_choices(
        anchor, bank, owner, records, global_choices, global_gains
    )
    global_summary, global_metrics = score_prediction(
        truth, global_prediction, valid, regimes
    )
    key = f"{mode}_global_family_oracle"
    rows.append(
        method_row(
            key,
            global_summary,
            anchor_summary,
            cell_summary,
            global_coverage,
        )
    )
    scores[key] = global_metrics["weighted"]

    crossfit_outputs: dict[str, np.ndarray] = {}
    crossfit_choices_by_key: dict[str, np.ndarray] = {}
    for radius in (None, *LOCAL_RADII):
        choices, gains, coverage = crossfit_family_choices(
            records,
            positions,
            fold_id,
            radius=radius,
            neighbors=LOCAL_NEIGHBORS,
            minimum_support=MIN_FAMILY_SUPPORT,
            scaled=True,
        )
        prediction = apply_record_choices(anchor, bank, owner, records, choices, gains)
        summary, metrics = score_prediction(truth, prediction, valid, regimes)
        radius_name = "global" if radius is None else f"r{int(radius)}"
        key = f"{mode}_crossfit_{radius_name}"
        rows.append(method_row(key, summary, anchor_summary, cell_summary, coverage))
        scores[key] = metrics["weighted"]
        crossfit_outputs[key] = prediction
        crossfit_choices_by_key[key] = choices
        print(
            f"v545 {key} weighted={summary['overall']['weighted']:.6f} "
            f"coverage={coverage['record_fraction']:.3f}",
            flush=True,
        )

    best_crossfit_key = max(
        crossfit_outputs,
        key=lambda value: float(np.mean(scores[value][valid])),
    )
    best_crossfit = crossfit_outputs[best_crossfit_key]

    family_ids = path_meta[f"family_{mode}"]
    recurrence = np.bincount(family_ids[family_ids >= 0].astype(np.int64))
    family_stats = {
        "mode": mode,
        "path_family_count": int(len(recurrence)),
        "families_at_least_3_path_occurrences": int(np.sum(recurrence >= 3)),
        "families_at_least_10_path_occurrences": int(np.sum(recurrence >= 10)),
        "families_at_least_50_path_occurrences": int(np.sum(recurrence >= 50)),
        "path_occurrence_fraction_in_recurrent_families": float(
            np.sum(recurrence[recurrence >= 3]) / max(np.sum(recurrence), 1)
        ),
        "best_crossfit_key": best_crossfit_key,
    }
    return rows, scores, scaled_prediction, best_crossfit, family_stats


def load_background() -> dict[str, object]:
    scene = np.load(SCENE)
    wall = np.log1p(scene["grid_vertical_count"].astype(np.float32))
    positive = wall[wall > 0]
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    return {
        "wall": wall,
        "extent": (
            x_min,
            x_min + wall.shape[1] * resolution,
            y_min,
            y_min + wall.shape[0] * resolution,
        ),
        "vmax": max(float(np.quantile(positive, 0.98)), 1.0),
    }


def decorate_map(axis: plt.Axes, background: dict[str, object]) -> None:
    axis.imshow(
        background["wall"],
        origin="lower",
        extent=background["extent"],
        cmap="Greys",
        vmin=0.0,
        vmax=background["vmax"],
        alpha=0.20,
        interpolation="nearest",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("横坐标（米）")
    axis.set_ylabel("纵坐标（米）")


def plot_ladder(
    methods: dict[str, dict[str, object]], anchor_score: float, cell_score: float
) -> None:
    preferred = [
        "anchor",
        "v543_cell_model",
        "cell_oracle",
        "grid_d3_a2",
        "exact_point_source",
        "exact_point_source_gain",
        "exact_global_family_oracle",
    ]
    crossfit = [key for key in methods if key.startswith("exact_crossfit_")]
    if crossfit:
        preferred.append(
            max(
                crossfit, key=lambda key: methods[key]["overall"]["overall"]["weighted"]
            )
        )
    labels = {
        "anchor": "物理锚点",
        "v543_cell_model": "v543严格OOF",
        "cell_oracle": "逐格oracle",
        "grid_d3_a2": "3×2连续谱块",
        "exact_point_source": "点内路径块",
        "exact_point_source_gain": "路径块+能量",
        "exact_global_family_oracle": "全局路径族oracle",
    }
    values = [
        float(methods[key]["overall"]["overall"]["weighted"]) for key in preferred
    ]
    names = [labels.get(key, "跨位置路径族OOF") for key in preferred]
    colors = [
        "#64748b",
        "#dc2626",
        "#f59e0b",
        "#7c3aed",
        "#2563eb",
        "#0f766e",
        "#16a34a",
        "#0891b2",
    ]
    figure, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    bars = axis.bar(np.arange(len(values)), values, color=colors[: len(values)])
    axis.bar_label(bars, fmt="%.4f", padding=3)
    axis.set_xticks(np.arange(len(values)), names, rotation=15)
    axis.set_ylim(max(0.65, min(values) - 0.04), 1.0)
    axis.set_ylabel("训练集严格OOF联合谱综合相似度")
    axis.grid(axis="y", alpha=0.25)
    axis.axhline(anchor_score, color="#64748b", linestyle="--", linewidth=1)
    axis.axhline(cell_score, color="#dc2626", linestyle=":", linewidth=1)
    axis.set_title(
        "从逐格碎片到跨位置路径族：物理约束逐级增加后的可实现能力",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(FIGURES["ladder"], bbox_inches="tight")
    plt.close(figure)


def plot_grid_curve(
    grid: dict[str, object], anchor_score: float, cell_score: float
) -> None:
    labels = []
    overall = []
    regime_values = {regime: [] for regime in REGIMES}
    for delay_size, angle_size in GRID_BLOCKS:
        key = f"d{delay_size}_a{angle_size}"
        labels.append(f"{delay_size}×{angle_size}")
        overall.append(float(grid[key]["overall"]["weighted"]))
        for regime in REGIMES:
            regime_values[regime].append(
                float(grid[key]["by_regime"][regime]["weighted"])
            )
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    axes[0].plot(labels, overall, marker="o", linewidth=2, color="#2563eb")
    axes[0].axhline(anchor_score, color="#64748b", linestyle="--", label="物理锚点")
    axes[0].axhline(cell_score, color="#dc2626", linestyle=":", label="逐格oracle")
    axes[0].set_ylabel("综合相似度")
    axes[0].set_xlabel("连续块大小（时延×角度bin）")
    axes[0].set_title("全体训练点", fontweight="bold")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    colors = ("#0f766e", "#eab308", "#f97316", "#dc2626")
    for regime, color in zip(REGIMES, colors):
        axes[1].plot(
            labels,
            regime_values[regime],
            marker="o",
            linewidth=1.8,
            color=color,
            label=REGIME_CN[regime],
        )
    axes[1].set_ylabel("综合相似度")
    axes[1].set_xlabel("连续块大小（时延×角度bin）")
    axes[1].set_title("不同传播区域", fontweight="bold")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle(
        "连续谱块共同选择同一候选后的上限敏感性",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(FIGURES["grid"], bbox_inches="tight")
    plt.close(figure)


def plot_family_sensitivity(methods: dict[str, dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    x = np.arange(len(IDENTITY_MODES), dtype=np.float64)
    values = []
    for mode in IDENTITY_MODES:
        values.append(
            [
                float(
                    methods[f"{mode}_point_source"]["overall"]["overall"]["weighted"]
                ),
                float(
                    methods[f"{mode}_point_source_gain"]["overall"]["overall"][
                        "weighted"
                    ]
                ),
                float(
                    methods[f"{mode}_global_family_oracle"]["overall"]["overall"][
                        "weighted"
                    ]
                ),
            ]
        )
    values_array = np.asarray(values)
    for index, (label, color) in enumerate(
        (
            ("点内路径块", "#7c3aed"),
            ("路径块+能量", "#2563eb"),
            ("全局共享路径族oracle", "#0f766e"),
        )
    ):
        axes[0].plot(
            x,
            values_array[:, index],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    axes[0].set_xticks(x, [IDENTITY_CN[mode] for mode in IDENTITY_MODES])
    axes[0].set_ylabel("综合相似度")
    axes[0].set_title("路径身份粒度", fontweight="bold")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    radius_labels = ("全局", *[f"{int(radius)}米" for radius in LOCAL_RADII])
    colors = ("#2563eb", "#f97316", "#16a34a")
    for mode, color in zip(IDENTITY_MODES, colors):
        keys = [f"{mode}_crossfit_global"] + [
            f"{mode}_crossfit_r{int(radius)}" for radius in LOCAL_RADII
        ]
        scores = [float(methods[key]["overall"]["overall"]["weighted"]) for key in keys]
        axes[1].plot(
            radius_labels,
            scores,
            marker="o",
            linewidth=2,
            color=color,
            label=IDENTITY_CN[mode],
        )
    axes[1].set_ylabel("严格OOF综合相似度")
    axes[1].set_title("同一路径族从其他训练位置选择候选", fontweight="bold")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle(
        "路径族身份与空间共享尺度敏感性",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(FIGURES["family"], bbox_inches="tight")
    plt.close(figure)


def plot_heavy_space(
    positions: np.ndarray,
    regimes: np.ndarray,
    valid: np.ndarray,
    anchor_score: np.ndarray,
    point_score: np.ndarray,
    crossfit_score: np.ndarray,
) -> None:
    heavy = valid & (regimes == "heavy")
    background = load_background()
    values = (
        (anchor_score, "物理锚点综合相似度", "viridis", False),
        (point_score - anchor_score, "点内路径块oracle增量", "magma", False),
        (crossfit_score - anchor_score, "跨位置路径族OOF增量", "RdBu_r", True),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    for axis, (value, title, cmap, symmetric) in zip(axes, values):
        decorate_map(axis, background)
        local = value[heavy]
        if symmetric:
            bound = max(float(np.quantile(np.abs(local), 0.98)), 1e-5)
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
            points = axis.scatter(
                positions[heavy, 0],
                positions[heavy, 1],
                c=local,
                s=20,
                cmap=cmap,
                norm=norm,
                linewidth=0,
                alpha=0.88,
            )
        else:
            low, high = np.quantile(local, (0.02, 0.98))
            points = axis.scatter(
                positions[heavy, 0],
                positions[heavy, 1],
                c=local,
                s=20,
                cmap=cmap,
                vmin=float(low),
                vmax=float(high),
                linewidth=0,
                alpha=0.88,
            )
        axis.set_title(title, fontweight="bold")
        plt.colorbar(points, ax=axis, fraction=0.045, pad=0.02)
    figure.suptitle(
        "1054个重遮挡训练点：路径块潜力与跨位置可迁移结果",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(FIGURES["space"], bbox_inches="tight")
    plt.close(figure)


def spectrum_db(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float64), 1e-30)
    return 10.0 * np.log10(value / np.max(value))


def plot_examples(
    truth: np.ndarray,
    anchor: np.ndarray,
    cell: np.ndarray,
    point_oracle: np.ndarray,
    crossfit: np.ndarray,
    positions: np.ndarray,
    regimes: np.ndarray,
    valid: np.ndarray,
    scores: dict[str, np.ndarray],
) -> None:
    heavy = np.flatnonzero(valid & (regimes == "heavy"))
    potential = scores["point"] - scores["anchor"]
    transfer = scores["crossfit"] - scores["anchor"]
    selected = [
        int(heavy[np.argmax(potential[heavy])]),
        int(heavy[np.argmax(transfer[heavy])]),
        int(heavy[np.argmin(transfer[heavy])]),
    ]
    arrays = (truth, anchor, cell, point_oracle, crossfit)
    titles = (
        "官方真实谱",
        "物理锚点",
        "逐格oracle",
        "点内路径块oracle",
        "跨位置路径族OOF",
    )
    figure, axes = plt.subplots(3, 5, figsize=(17, 10.5), constrained_layout=True)
    for row, point in enumerate(selected):
        for column, (array, title) in enumerate(zip(arrays, titles)):
            image = axes[row, column].imshow(
                spectrum_db(array[point].reshape(HEIGHT, WIDTH)),
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=-35.0,
                vmax=0.0,
                interpolation="nearest",
            )
            if row == 0:
                axes[row, column].set_title(title, fontweight="bold")
            if row == 2:
                axes[row, column].set_xlabel("水平角度bin")
            if column == 0:
                axes[row, column].set_ylabel(
                    f"点{point} ({positions[point,0]:.1f}, {positions[point,1]:.1f})\n时延bin"
                )
            plt.colorbar(image, ax=axes[row, column], fraction=0.045, pad=0.02)
    figure.suptitle(
        "典型重遮挡训练点：逐格高上限能否保留为完整路径结构",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(FIGURES["examples"], bbox_inches="tight")
    plt.close(figure)


def plot_coverage(
    family_stats: dict[str, dict[str, object]], methods: dict[str, dict[str, object]]
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    x = np.arange(len(IDENTITY_MODES), dtype=np.float64)
    totals = [family_stats[mode]["path_family_count"] for mode in IDENTITY_MODES]
    recurrent3 = [
        family_stats[mode]["families_at_least_3_path_occurrences"]
        for mode in IDENTITY_MODES
    ]
    recurrent10 = [
        family_stats[mode]["families_at_least_10_path_occurrences"]
        for mode in IDENTITY_MODES
    ]
    axes[0].bar(x - 0.22, totals, 0.22, label="全部路径族", color="#64748b")
    axes[0].bar(x, recurrent3, 0.22, label="至少出现3次", color="#2563eb")
    axes[0].bar(x + 0.22, recurrent10, 0.22, label="至少出现10次", color="#16a34a")
    axes[0].set_xticks(x, [IDENTITY_CN[mode] for mode in IDENTITY_MODES])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("路径族数量（对数坐标）")
    axes[0].set_title("路径族在训练点间的复现频率", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.22)

    radius_labels = ("全局", *[f"{int(radius)}米" for radius in LOCAL_RADII])
    for mode, color in zip(IDENTITY_MODES, ("#2563eb", "#f97316", "#16a34a")):
        keys = [f"{mode}_crossfit_global"] + [
            f"{mode}_crossfit_r{int(radius)}" for radius in LOCAL_RADII
        ]
        coverage = [methods[key]["coverage"]["record_fraction"] for key in keys]
        axes[1].plot(
            radius_labels,
            coverage,
            marker="o",
            linewidth=2,
            color=color,
            label=IDENTITY_CN[mode],
        )
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("可从其他折获得监督的路径块比例")
    axes[1].set_title("严格交叉拟合覆盖率", fontweight="bold")
    axes[1].grid(alpha=0.22)
    axes[1].legend()
    figure.suptitle(
        "路径族是否具有足够的跨位置重复证据",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(FIGURES["coverage"], bbox_inches="tight")
    plt.close(figure)


def write_summary_csv(methods: dict[str, dict[str, object]]) -> None:
    fields = (
        "方法",
        "综合相似度",
        "相对物理锚点增量",
        "逐格oracle增量保留比例",
        "无遮挡",
        "轻遮挡",
        "中遮挡",
        "重遮挡",
        "路径块覆盖率",
    )
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, row in methods.items():
            summary = row["overall"]
            coverage = row.get("coverage", {})
            writer.writerow(
                {
                    "方法": key,
                    "综合相似度": f"{summary['overall']['weighted']:.9f}",
                    "相对物理锚点增量": f"{row['gain_vs_anchor']:.9f}",
                    "逐格oracle增量保留比例": f"{row['cell_oracle_gain_retention']:.9f}",
                    "无遮挡": f"{summary['by_regime']['clear']['weighted']:.9f}",
                    "轻遮挡": f"{summary['by_regime']['light']['weighted']:.9f}",
                    "中遮挡": f"{summary['by_regime']['medium']['weighted']:.9f}",
                    "重遮挡": f"{summary['by_regime']['heavy']['weighted']:.9f}",
                    "路径块覆盖率": f"{coverage.get('record_fraction', 1.0):.9f}",
                }
            )


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    validate_inputs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    oof = np.load(OOF_SOURCE)
    truth = oof["truth"].astype(np.float32)
    anchor = oof["anchor"].astype(np.float32)
    cell_model = oof["calibrated"].astype(np.float32)
    cell_oracle = oof["cell_oracle"].astype(np.float32)
    regimes = oof["regimes"].astype("U8")
    positions = np.load(TRAIN_POS).astype(np.float32)
    power = np.load(TRAIN_POWER).astype(np.float64)
    valid = power != 0.0
    fold_id = oof["fold_id"].astype(np.int16)

    bank, source_names, bank_meta = build_oof_bank(args.rebuild_bank)
    path_meta = build_path_metadata(args.rebuild_path_meta)
    anchor_summary, anchor_metrics = score_prediction(truth, anchor, valid, regimes)
    model_summary, model_metrics = score_prediction(truth, cell_model, valid, regimes)
    cell_summary, cell_metrics = score_prediction(truth, cell_oracle, valid, regimes)
    methods: dict[str, dict[str, object]] = {
        "anchor": method_row("anchor", anchor_summary, anchor_summary, cell_summary),
        "v543_cell_model": method_row(
            "v543_cell_model", model_summary, anchor_summary, cell_summary
        ),
        "cell_oracle": method_row(
            "cell_oracle", cell_summary, anchor_summary, cell_summary
        ),
    }
    all_scores: dict[str, np.ndarray] = {
        "anchor": anchor_metrics["weighted"],
        "v543_cell_model": model_metrics["weighted"],
        "cell_oracle": cell_metrics["weighted"],
    }

    grid_summaries, grid_scores = compute_grid_curve(truth, bank, valid, regimes)
    for key, summary in grid_summaries.items():
        method_key = f"grid_{key}"
        methods[method_key] = method_row(
            method_key, summary, anchor_summary, cell_summary
        )
        all_scores[method_key] = grid_scores[key]

    family_stats: dict[str, dict[str, object]] = {}
    exact_point_prediction: np.ndarray | None = None
    exact_best_crossfit: np.ndarray | None = None
    for mode in IDENTITY_MODES:
        rows, scores, point_prediction, best_crossfit, stats = evaluate_family_mode(
            mode,
            truth,
            anchor,
            bank,
            path_meta,
            valid,
            regimes,
            positions,
            fold_id,
            anchor_summary,
            cell_summary,
        )
        for row in rows:
            methods[str(row["name"])] = row
        all_scores.update(scores)
        family_stats[mode] = stats
        if mode == "exact":
            exact_point_prediction = point_prediction
            exact_best_crossfit = best_crossfit

    if exact_point_prediction is None or exact_best_crossfit is None:
        raise RuntimeError("missing exact path-family outputs")
    np.save(POINT_ORACLE_CACHE, exact_point_prediction.astype(np.float32))
    np.save(CROSSFIT_CACHE, exact_best_crossfit.astype(np.float32))
    np.savez_compressed(
        SCORE_CACHE,
        valid=valid,
        regimes=regimes,
        **{key: value.astype(np.float32) for key, value in all_scores.items()},
    )

    plot_ladder(
        methods,
        float(anchor_summary["overall"]["weighted"]),
        float(cell_summary["overall"]["weighted"]),
    )
    plot_grid_curve(
        grid_summaries,
        float(anchor_summary["overall"]["weighted"]),
        float(cell_summary["overall"]["weighted"]),
    )
    plot_family_sensitivity(methods)
    exact_best_key = str(family_stats["exact"]["best_crossfit_key"])
    plot_heavy_space(
        positions,
        regimes,
        valid,
        all_scores["anchor"],
        all_scores["exact_point_source_gain"],
        all_scores[exact_best_key],
    )
    plot_examples(
        truth,
        anchor,
        cell_oracle,
        exact_point_prediction,
        exact_best_crossfit,
        positions,
        regimes,
        valid,
        {
            "anchor": all_scores["anchor"],
            "point": all_scores["exact_point_source_gain"],
            "crossfit": all_scores[exact_best_key],
        },
    )
    plot_coverage(family_stats, methods)
    write_summary_csv(methods)

    point_score = float(
        methods["exact_point_source_gain"]["overall"]["overall"]["weighted"]
    )
    best_crossfit_score = float(
        methods[exact_best_key]["overall"]["overall"]["weighted"]
    )
    anchor_score = float(anchor_summary["overall"]["weighted"])
    model_score = float(model_summary["overall"]["weighted"])
    cell_score = float(cell_summary["overall"]["weighted"])
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The high v543 cell oracle is useful only if its gain survives contiguous "
            "path blocks and path identities shared across spatially neighboring points."
        ),
        "scope": {
            "train_points": 2000,
            "nonzero_scored_points": int(np.sum(valid)),
            "exact_zero_marked_not_scored": int(np.sum(~valid)),
            "test_coordinates_or_predictions_used": False,
            "oof_candidate_calibration": True,
        },
        "parameters": {
            "sources": source_names,
            "grid_blocks": [list(value) for value in GRID_BLOCKS],
            "identity_modes": list(IDENTITY_MODES),
            "local_radii_m": list(LOCAL_RADII),
            "local_neighbors": LOCAL_NEIGHBORS,
            "minimum_family_support": MIN_FAMILY_SUPPORT,
            "gain_range": [GAIN_MIN, GAIN_MAX],
            "path_block_owner": (
                "strongest v475 Sionna path atom per delay-angle cell; path atoms use "
                "continuous delay fingerprints, TX horizontal fingerprints, and simulated mass"
            ),
        },
        "validation": {
            "bank": bank_meta,
            "path_atom_to_saved_joint_cosine": summarize(
                path_meta["atom_cosine"][valid]
            ),
            "mean_path_count": float(np.mean(path_meta["path_count"][valid])),
        },
        "methods": methods,
        "family_stats": family_stats,
        "decision": {
            "anchor_score": anchor_score,
            "v543_cell_model_score": model_score,
            "cell_oracle_score": cell_score,
            "exact_point_path_block_score": point_score,
            "exact_point_retained_cell_gain": (point_score - anchor_score)
            / max(cell_score - anchor_score, 1e-12),
            "best_exact_crossfit_key": exact_best_key,
            "best_exact_crossfit_score": best_crossfit_score,
            "best_exact_crossfit_gain": best_crossfit_score - anchor_score,
            "best_exact_crossfit_gain_vs_v543": best_crossfit_score - model_score,
            "high_cell_oracle_is_path_coherent": bool(
                (point_score - anchor_score) >= 0.5 * (cell_score - anchor_score)
            ),
            "path_family_selection_transfers_without_query_truth": bool(
                best_crossfit_score > anchor_score
            ),
            "path_family_crossfit_exceeds_v543": bool(
                best_crossfit_score > model_score
            ),
        },
        "outputs": {
            "candidate_bank": PATHS.relative(BANK_CACHE),
            "path_geometry": PATHS.relative(PATH_META),
            "scores": PATHS.relative(SCORE_CACHE),
            "point_path_block_oracle": PATHS.relative(POINT_ORACLE_CACHE),
            "best_crossfit": PATHS.relative(CROSSFIT_CACHE),
            "summary_csv": PATHS.relative(SUMMARY_CSV),
            "figures": {key: PATHS.relative(path) for key, path in FIGURES.items()},
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
