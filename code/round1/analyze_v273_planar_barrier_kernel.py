#!/usr/bin/env python3
"""Nested holdout audit of a 2.5D barrier-aware planar propagation kernel."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import analyze_v254_route_topology_spectral_field as v254
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 273
VERSION = "v273_planar_barrier_kernel"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
FEATURE_CACHE = OUT_DIR / "v273_planar_barrier_features.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
FOOTPRINT_CACHE = PATHS.cache_dir(272) / "v272_extruded_footprint.npz"

BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
OUTER_SEEDS = (273, 1273, 2273, 3273, 4273)
INNER_OFFSETS = (10_000, 20_000)
OUTER_COUNT = 320
INNER_COUNT = 240
SEARCH_K = 72
SAMPLE_STEP = 0.5
MIN_DELTA = 5e-4


@dataclass(frozen=True)
class Kernel:
    k: int
    power: float
    segment_penalty: float
    signature_penalty: float
    height_scale: float

    @property
    def name(self) -> str:
        return (
            f"k{self.k}_p{self.power:g}_seg{self.segment_penalty:g}_"
            f"sig{self.signature_penalty:g}_h{self.height_scale:g}"
        )

    @property
    def barrier_aware(self) -> bool:
        return self.segment_penalty > 0.0 or self.signature_penalty > 0.0


@dataclass
class PairFeatures:
    indices: np.ndarray
    distance: np.ndarray
    segment_barrier: np.ndarray
    segment_height: np.ndarray
    signature_distance: np.ndarray


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


def weighted_bank(
    bank: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    reshape = weights.shape + (1,) * (bank.ndim - 1)
    return normalize(
        np.sum(
            bank[indices] * weights.reshape(reshape),
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)
    )


def matched_subset(
    candidate_ids: np.ndarray,
    train_xy: np.ndarray,
    test_xy: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    candidate_xy = train_xy[candidate_ids]
    tree = cKDTree(candidate_xy)
    candidate_distance = tree.query(candidate_xy, k=2)[0][:, 1]
    test_distance = tree.query(test_xy, k=1)[0]
    count = min(count, len(candidate_ids) // 3)
    targets = np.quantile(test_distance, (np.arange(count) + 0.5) / count)
    order = np.argsort(candidate_distance)
    sorted_distance = candidate_distance[order]
    used = np.zeros(len(candidate_ids), dtype=bool)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for target in targets:
        center = int(np.searchsorted(sorted_distance, target))
        lo = max(center - 80, 0)
        hi = min(center + 81, len(order))
        local = order[lo:hi]
        local = local[~used[local]]
        if len(local) == 0:
            local = np.flatnonzero(~used)
        error = np.abs(candidate_distance[local] - target)
        shortlist_order = np.argsort(error)[: min(10, len(local))]
        shortlist = local[shortlist_order]
        shortlist_error = error[shortlist_order]
        scale = max(float(np.median(shortlist_error)), 1e-6)
        probability = np.exp(-shortlist_error / scale)
        probability /= probability.sum()
        selected = int(rng.choice(shortlist, p=probability))
        used[selected] = True
        chosen.append(int(candidate_ids[selected]))
    return np.sort(np.asarray(chosen, dtype=np.int64))


def grid_indices(
    xy: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.floor((xy[..., 0] - x_min) / resolution).astype(np.int64)
    iy = np.floor((xy[..., 1] - y_min) / resolution).astype(np.int64)
    valid = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    return ix, iy, valid


def bs_component_signatures(
    positions: np.ndarray,
    labels: np.ndarray,
    z_max: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    component_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    signatures = np.zeros((len(positions), component_count), dtype=bool)
    blocker_count = np.zeros(len(positions), dtype=np.int16)
    for row, position in enumerate(positions):
        length = float(np.linalg.norm(position[:2] - BS[:2]))
        steps = max(int(np.ceil(length / SAMPLE_STEP)), 2)
        t = np.linspace(0.01, 0.99, steps, dtype=np.float64)
        xy = BS[None, :2] + t[:, None] * (position[None, :2] - BS[None, :2])
        ray_z = BS[2] + t * (position[2] - BS[2])
        ix, iy, valid = grid_indices(
            xy, x_min, y_min, resolution, labels.shape
        )
        blocked = np.zeros(steps, dtype=bool)
        blocked[valid] = (
            labels[iy[valid], ix[valid]] > 0
        ) & (
            np.nan_to_num(z_max[iy[valid], ix[valid]], nan=-np.inf)
            >= ray_z[valid] + 0.2
        )
        component = np.zeros(steps, dtype=np.int32)
        component[valid & blocked] = labels[iy[valid & blocked], ix[valid & blocked]]
        ids = np.unique(component[component > 0])
        ids = ids[ids <= component_count]
        signatures[row, ids - 1] = True
        blocker_count[row] = len(ids)
    return signatures, blocker_count


def segment_profile(
    start: np.ndarray,
    end: np.ndarray,
    footprint: np.ndarray,
    z_max: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
) -> tuple[float, float]:
    length = float(np.linalg.norm(end - start))
    steps = max(int(np.ceil(length / SAMPLE_STEP)), 2)
    t = np.linspace(0.03, 0.97, steps, dtype=np.float64)
    xy = start[None, :] + t[:, None] * (end[None, :] - start[None, :])
    ix, iy, valid = grid_indices(xy, x_min, y_min, resolution, footprint.shape)
    occupied = np.zeros(steps, dtype=bool)
    occupied[valid] = footprint[iy[valid], ix[valid]]
    runs = float(np.sum(occupied & ~np.r_[False, occupied[:-1]]))
    wall_length = float(np.sum(occupied) * length / max(steps, 1))
    barrier = runs + 0.25 * wall_length
    if np.any(occupied & valid):
        sampled_height = z_max[iy[occupied & valid], ix[occupied & valid]]
        sampled_height = sampled_height[np.isfinite(sampled_height)]
        height = float(np.max(sampled_height)) if len(sampled_height) else 0.0
    else:
        height = 0.0
    return barrier, max(height, 0.0)


def pair_features(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    positions: np.ndarray,
    signatures: np.ndarray,
    footprint: np.ndarray,
    z_max: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
) -> PairFeatures:
    take = min(SEARCH_K, len(bank_ids))
    distance, local = cKDTree(positions[bank_ids, :2]).query(
        positions[query_ids, :2], k=take
    )
    if take == 1:
        distance = distance[:, None]
        local = local[:, None]
    absolute = bank_ids[local]
    barrier = np.zeros_like(distance, dtype=np.float32)
    height = np.zeros_like(distance, dtype=np.float32)
    for row, query_id in enumerate(query_ids):
        for col, bank_id in enumerate(absolute[row]):
            barrier[row, col], height[row, col] = segment_profile(
                positions[query_id, :2],
                positions[bank_id, :2],
                footprint,
                z_max,
                x_min,
                y_min,
                resolution,
            )
    query_signature = signatures[query_ids, None, :]
    bank_signature = signatures[absolute]
    difference = np.sum(query_signature != bank_signature, axis=2)
    union = np.sum(query_signature | bank_signature, axis=2)
    signature_distance = difference / np.maximum(union, 1)
    return PairFeatures(
        indices=absolute.astype(np.int64),
        distance=np.asarray(distance, dtype=np.float32),
        segment_barrier=barrier,
        segment_height=height,
        signature_distance=signature_distance.astype(np.float32),
    )


def predict(
    features: PairFeatures,
    kernel: Kernel,
    sh: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height_factor = 1.0 + kernel.height_scale * np.clip(
        features.segment_height / 25.0, 0.0, 2.0
    )
    effective = (
        features.distance
        + kernel.segment_penalty * features.segment_barrier * height_factor
        + kernel.signature_penalty * features.signature_distance
    )
    order = np.argsort(effective, axis=1)[:, : kernel.k]
    indices = np.take_along_axis(features.indices, order, axis=1)
    selected_distance = np.take_along_axis(effective, order, axis=1)
    weight = 1.0 / np.maximum(selected_distance, 0.05) ** kernel.power
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-20)
    return weighted_bank(sh, indices, weight), weighted_bank(d, indices, weight)


def kernels() -> list[Kernel]:
    output: list[Kernel] = []
    for k in (8, 12, 16, 24):
        for power in (1.5, 2.0):
            output.append(Kernel(k, power, 0.0, 0.0, 0.0))
    feature_weights = (
        (0.0, 4.0, 0.0),
        (0.0, 12.0, 0.0),
        (0.0, 24.0, 0.0),
        (4.0, 0.0, 0.0),
        (12.0, 0.0, 0.0),
        (24.0, 0.0, 0.0),
        (4.0, 4.0, 0.0),
        (8.0, 12.0, 0.0),
        (12.0, 24.0, 0.0),
        (24.0, 12.0, 0.0),
        (4.0, 4.0, 1.0),
        (8.0, 12.0, 1.0),
        (12.0, 24.0, 1.0),
        (24.0, 12.0, 1.0),
    )
    for k in (8, 12, 16):
        for power in (1.5, 2.0):
            for segment_penalty, signature_penalty, height_scale in feature_weights:
                output.append(
                    Kernel(
                        k,
                        power,
                        segment_penalty,
                        signature_penalty,
                        height_scale,
                    )
                )
    return output


def select_kernel(
    candidate_kernels: list[Kernel],
    feature_sets: list[PairFeatures],
    validation_sets: list[np.ndarray],
    sh: np.ndarray,
    d: np.ndarray,
    barrier_aware: bool,
) -> tuple[Kernel, dict[str, float]]:
    selected = [kernel for kernel in candidate_kernels if kernel.barrier_aware == barrier_aware]
    scores: dict[str, list[dict[str, float]]] = {kernel.name: [] for kernel in selected}
    for features, validation_ids in zip(feature_sets, validation_sets):
        for kernel in selected:
            pred_sh, pred_d = predict(features, kernel, sh, d)
            scores[kernel.name].append(
                metrics(pred_sh, pred_d, sh[validation_ids], d[validation_ids])
            )
    best = max(
        selected,
        key=lambda kernel: np.mean(
            [row["weighted"] for row in scores[kernel.name]]
        ),
    )
    mean = {
        key: float(np.mean([row[key] for row in scores[best.name]]))
        for key in ("sh", "d", "weighted")
    }
    return best, mean


def main() -> None:
    for path in (TARGET_CACHE, SCENE_CACHE, FOOTPRINT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    targets = np.load(TARGET_CACHE)
    sh = targets["sh"].astype(np.float32)
    d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    scene = np.load(SCENE_CACHE)
    footprint_data = np.load(FOOTPRINT_CACHE)
    labels = scene["component_labels"].astype(np.int32)
    z_max = scene["grid_z_max"].astype(np.float32)
    footprint = footprint_data["footprint"].astype(bool)
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    component_count = int(np.max(labels))
    all_pos = np.vstack([train_pos, test_pos])
    signatures, blocker_count = bs_component_signatures(
        all_pos,
        labels,
        z_max,
        x_min,
        y_min,
        resolution,
        component_count,
    )
    np.savez_compressed(
        FEATURE_CACHE,
        train_signature=signatures[: len(train_pos)],
        test_signature=signatures[len(train_pos) :],
        train_blocker_count=blocker_count[: len(train_pos)],
        test_blocker_count=blocker_count[len(train_pos) :],
    )

    candidate_kernels = kernels()
    outer_rows: list[dict[str, object]] = []
    for outer_index, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_feature_sets: list[PairFeatures] = []
        inner_validation_sets: list[np.ndarray] = []
        for offset in INNER_OFFSETS:
            inner_ids = matched_subset(
                outer_fit,
                train_pos[:, :2],
                test_pos[:, :2],
                INNER_COUNT,
                seed + offset,
            )
            inner_fit = np.setdiff1d(outer_fit, inner_ids)
            inner_feature_sets.append(
                pair_features(
                    inner_ids,
                    inner_fit,
                    train_pos,
                    signatures[: len(train_pos)],
                    footprint,
                    z_max,
                    x_min,
                    y_min,
                    resolution,
                )
            )
            inner_validation_sets.append(inner_ids)
        baseline, baseline_inner = select_kernel(
            candidate_kernels,
            inner_feature_sets,
            inner_validation_sets,
            sh,
            d,
            False,
        )
        barrier, barrier_inner = select_kernel(
            candidate_kernels,
            inner_feature_sets,
            inner_validation_sets,
            sh,
            d,
            True,
        )
        outer_features = pair_features(
            outer_ids,
            outer_fit,
            train_pos,
            signatures[: len(train_pos)],
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        baseline_pred = predict(outer_features, baseline, sh, d)
        barrier_pred = predict(outer_features, barrier, sh, d)
        baseline_outer = metrics(
            baseline_pred[0], baseline_pred[1], sh[outer_ids], d[outer_ids]
        )
        barrier_outer = metrics(
            barrier_pred[0], barrier_pred[1], sh[outer_ids], d[outer_ids]
        )
        deploy_barrier = bool(
            barrier_inner["weighted"] - baseline_inner["weighted"] >= MIN_DELTA
        )
        deployed_outer = barrier_outer if deploy_barrier else baseline_outer
        delta = {
            key: barrier_outer[key] - baseline_outer[key]
            for key in ("sh", "d", "weighted")
        }
        deployed_delta = {
            key: deployed_outer[key] - baseline_outer[key]
            for key in ("sh", "d", "weighted")
        }
        outer_rows.append(
            {
                "fold": outer_index,
                "seed": seed,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "selected_baseline": baseline.name,
                "selected_barrier": barrier.name,
                "inner_baseline": baseline_inner,
                "inner_barrier": barrier_inner,
                "outer_baseline": baseline_outer,
                "outer_barrier": barrier_outer,
                "outer_delta": delta,
                "deploy_barrier": deploy_barrier,
                "outer_deployed": deployed_outer,
                "outer_deployed_delta": deployed_delta,
                "nearest_candidate_barrier_fraction": float(
                    np.mean(outer_features.segment_barrier[:, 0] > 0.0)
                ),
                "candidate_barrier_fraction": float(
                    np.mean(outer_features.segment_barrier > 0.0)
                ),
                "candidate_signature_difference": summarize(
                    outer_features.signature_distance
                ),
            }
        )
        print(
            f"v273 outer {outer_index + 1}/{len(OUTER_SEEDS)} "
            f"baseline={baseline_outer['weighted']:.6f} "
            f"barrier={barrier_outer['weighted']:.6f} "
            f"raw_delta={delta['weighted']:+.6f} "
            f"deploy={deploy_barrier} deployed_delta={deployed_delta['weighted']:+.6f}",
            flush=True,
        )

    delta_weighted = np.asarray(
        [row["outer_delta"]["weighted"] for row in outer_rows], dtype=np.float64
    )
    delta_sh = np.asarray(
        [row["outer_delta"]["sh"] for row in outer_rows], dtype=np.float64
    )
    delta_d = np.asarray(
        [row["outer_delta"]["d"] for row in outer_rows], dtype=np.float64
    )
    deployed_delta_weighted = np.asarray(
        [row["outer_deployed_delta"]["weighted"] for row in outer_rows],
        dtype=np.float64,
    )
    deployed_delta_sh = np.asarray(
        [row["outer_deployed_delta"]["sh"] for row in outer_rows],
        dtype=np.float64,
    )
    deployed_delta_d = np.asarray(
        [row["outer_deployed_delta"]["d"] for row in outer_rows],
        dtype=np.float64,
    )
    selected = bool(
        np.mean(deployed_delta_weighted) >= MIN_DELTA
        and np.sum(deployed_delta_weighted > 0.0) >= 3
        and np.mean(deployed_delta_sh) > 0.0
        and np.mean(deployed_delta_d) > 0.0
    )
    report = {
        "method": VERSION,
        "status": (
            "official_holdout_passed_next_train_neural_field"
            if selected
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Because every UE lies on z=1.5, the channel is a 2D field whose discontinuities "
            "are aligned with point-cloud wall barriers and BS-to-UE occlusion components."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H and PDP targets",
                "official train/test XY coordinates",
                "official point-cloud wall/component/height grids",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher outputs",
                "local replay truth",
                "hidden labels",
                "RT MIMO H or path coefficients",
            ],
            "local_truth_usage": "not read",
        },
        "parameters": {
            "outer_seeds": list(OUTER_SEEDS),
            "inner_offsets": list(INNER_OFFSETS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "search_k": SEARCH_K,
            "sample_step_m": SAMPLE_STEP,
            "kernel_count": len(candidate_kernels),
            "minimum_delta": MIN_DELTA,
        },
        "planarity": {
            "train_z_unique": np.unique(train_pos[:, 2]).tolist(),
            "test_z_unique": np.unique(test_pos[:, 2]).tolist(),
            "coordinate_singular_values": np.linalg.svd(
                all_pos - np.mean(all_pos, axis=0, keepdims=True),
                compute_uv=False,
            ).tolist(),
        },
        "propagation_signature": {
            "component_count": component_count,
            "train_blocker_count": summarize(blocker_count[: len(train_pos)]),
            "test_blocker_count": summarize(blocker_count[len(train_pos) :]),
            "train_clear_fraction": float(
                np.mean(blocker_count[: len(train_pos)] == 0)
            ),
            "test_clear_fraction": float(
                np.mean(blocker_count[len(train_pos) :] == 0)
            ),
        },
        "outer_folds": outer_rows,
        "selection": {
            "mean_delta": {
                "sh": float(np.mean(delta_sh)),
                "d": float(np.mean(delta_d)),
                "weighted": float(np.mean(delta_weighted)),
            },
            "mean_deployed_delta": {
                "sh": float(np.mean(deployed_delta_sh)),
                "d": float(np.mean(deployed_delta_d)),
                "weighted": float(np.mean(deployed_delta_weighted)),
            },
            "weighted_wins": int(np.sum(delta_weighted > 0.0)),
            "sh_wins": int(np.sum(delta_sh > 0.0)),
            "d_wins": int(np.sum(delta_d > 0.0)),
            "deployed_weighted_wins": int(np.sum(deployed_delta_weighted > 0.0)),
            "deployed_fold_count": int(
                np.sum([row["deploy_barrier"] for row in outer_rows])
            ),
            "selected": selected,
        },
        "outputs": {
            "features": PATHS.relative(FEATURE_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
