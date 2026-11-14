#!/usr/bin/env python3
"""v294: heteroscedastic planar Gaussian-process spectral field.

Official train H is treated as a noisy observation of a clean 2-D propagation
field. Local ordinary kriging uses a Matern covariance, per-observation noise
estimated from even/odd antenna PDP agreement, official-pointcloud blocker
signatures, and depth-16 path-state summaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v274_planar_barrier_attention as v274
import analyze_v292_highdepth_path_weight_transformer as v292
import analyze_v293_interaction_physical_path_law as v293
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 294
VERSION = "v294_heteroscedastic_planar_gp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v294_planar_gp_targets.npz"
OUTPUT = OUT_DIR / "v294_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
RELIABILITY_CACHE = PATHS.cache_dir(237) / "v237_denoise_diagnostics.npz"
BARRIER_CACHE = PATHS.cache_dir(273) / "v273_planar_barrier_features.npz"
PATH_CACHE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
INTERACTION_CACHE = PATHS.cache_dir(293) / "v293_interaction_features.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"

LOCAL_GATE = 0.645604
OUTER_SEEDS = (294, 1294, 2294, 3294, 4294)
OUTER_COUNT = 320
INNER_COUNT = 240
NEIGHBORS = 32
BASELINE_K = 16
MIN_OFFICIAL_DELTA = 5e-4
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)


@dataclass(frozen=True)
class Kernel:
    length_scale: float
    ridge: float
    noise_scale: float
    signature_scale: float
    path_scale: float
    positive_weights: bool

    @property
    def name(self) -> str:
        return (
            f"l{self.length_scale:g}_r{self.ridge:g}_n{self.noise_scale:g}_"
            f"sig{self.signature_scale:g}_path{self.path_scale:g}_"
            f"{'pos' if self.positive_weights else 'signed'}"
        )


def kernels() -> list[Kernel]:
    output: list[Kernel] = []
    for length in (2.5, 4.0, 6.5, 10.0, 16.0):
        for ridge in (0.005, 0.02, 0.08):
            output.append(Kernel(length, ridge, 0.0, 0.0, 0.0, True))
    for length in (4.0, 6.5, 10.0):
        for noise in (1.5, 5.0, 12.0):
            output.append(Kernel(length, 0.02, noise, 0.0, 0.0, True))
    for length in (4.0, 6.5, 10.0):
        output.extend(
            [
                Kernel(length, 0.02, 5.0, 0.75, 0.0, True),
                Kernel(length, 0.02, 5.0, 1.75, 0.0, True),
                Kernel(length, 0.02, 5.0, 0.0, 0.75, True),
                Kernel(length, 0.02, 5.0, 0.0, 1.75, True),
                Kernel(length, 0.02, 5.0, 0.75, 0.75, True),
                Kernel(length, 0.02, 5.0, 1.75, 1.75, True),
                Kernel(length, 0.02, 5.0, 0.75, 0.75, False),
            ]
        )
    return output


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
    prediction_sh: np.ndarray,
    prediction_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(prediction_sh, truth_sh)))
    d = float(np.mean(row_cos(prediction_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def weighted_bank(bank: np.ndarray, indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
    reshape = weight.shape + (1,) * (bank.ndim - 1)
    return normalize(
        np.sum(
            bank[indices] * weight.reshape(reshape), axis=1, dtype=np.float64
        ).astype(np.float32)
    )


def nearest(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    query_pos: np.ndarray,
    bank_pos: np.ndarray,
    count: int,
    exclude_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    extra = 1 if exclude_self else 0
    distance, local = cKDTree(bank_pos[bank_ids, :2]).query(
        query_pos[query_ids, :2], k=min(count + extra, len(bank_ids))
    )
    if distance.ndim == 1:
        distance = distance[:, None]
        local = local[:, None]
    absolute_all = bank_ids[local]
    if not exclude_self:
        return absolute_all[:, :count], distance[:, :count].astype(np.float32)
    absolute = np.empty((len(query_ids), count), dtype=np.int64)
    kept_distance = np.empty((len(query_ids), count), dtype=np.float32)
    for row, query_id in enumerate(query_ids):
        keep = absolute_all[row] != query_id
        absolute[row] = absolute_all[row, keep][:count]
        kept_distance[row] = distance[row, keep][:count]
    return absolute, kept_distance


def knn_prediction(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    query_pos: np.ndarray,
    bank_pos: np.ndarray,
    bank: np.ndarray,
    exclude_self: bool,
) -> np.ndarray:
    indices, distance = nearest(
        query_ids, bank_ids, query_pos, bank_pos, BASELINE_K, exclude_self
    )
    weight = 1.0 / np.maximum(distance, 0.05) ** 2
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-20)
    return weighted_bank(bank, indices, weight.astype(np.float32))


def jaccard_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    difference = np.sum(a != b, axis=-1)
    union = np.sum(a | b, axis=-1)
    return (difference / np.maximum(union, 1)).astype(np.float32)


@dataclass
class Neighborhood:
    indices: np.ndarray
    query_distance: np.ndarray
    pair_distance: np.ndarray
    query_signature_distance: np.ndarray
    pair_signature_distance: np.ndarray
    query_path_distance: np.ndarray
    pair_path_distance: np.ndarray


def neighborhood(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    query_pos: np.ndarray,
    bank_pos: np.ndarray,
    query_signature: np.ndarray,
    bank_signature: np.ndarray,
    query_path: np.ndarray,
    bank_path: np.ndarray,
    exclude_self: bool,
) -> Neighborhood:
    indices, query_distance = nearest(
        query_ids,
        bank_ids,
        query_pos,
        bank_pos,
        NEIGHBORS,
        exclude_self,
    )
    selected_pos = bank_pos[indices, :2]
    pair_distance = np.linalg.norm(
        selected_pos[:, :, None, :] - selected_pos[:, None, :, :], axis=3
    ).astype(np.float32)
    selected_signature = bank_signature[indices]
    query_signature_distance = jaccard_distance(
        query_signature[query_ids, None, :], selected_signature
    )
    pair_signature_distance = jaccard_distance(
        selected_signature[:, :, None, :], selected_signature[:, None, :, :]
    )
    selected_path = bank_path[indices]
    query_path_distance = np.mean(
        np.abs(query_path[query_ids, None, :] - selected_path), axis=2
    ).astype(np.float32)
    pair_path_distance = np.mean(
        np.abs(selected_path[:, :, None, :] - selected_path[:, None, :, :]),
        axis=3,
    ).astype(np.float32)
    return Neighborhood(
        indices,
        query_distance,
        pair_distance,
        query_signature_distance,
        pair_signature_distance,
        query_path_distance,
        pair_path_distance,
    )


def matern32(distance: np.ndarray, length_scale: float) -> np.ndarray:
    scaled = np.sqrt(3.0) * distance / max(length_scale, 1e-3)
    return ((1.0 + scaled) * np.exp(-scaled)).astype(np.float64)


def kriging_weights(
    data: Neighborhood,
    kernel: Kernel,
    observation_noise: np.ndarray,
) -> np.ndarray:
    pair = matern32(data.pair_distance, kernel.length_scale)
    query = matern32(data.query_distance, kernel.length_scale)
    if kernel.signature_scale > 0.0:
        pair *= np.exp(
            -kernel.signature_scale * data.pair_signature_distance
        )
        query *= np.exp(
            -kernel.signature_scale * data.query_signature_distance
        )
    if kernel.path_scale > 0.0:
        pair *= np.exp(-kernel.path_scale * data.pair_path_distance)
        query *= np.exp(-kernel.path_scale * data.query_path_distance)
    count, neighbors = data.indices.shape
    matrix = pair.copy()
    diagonal = kernel.ridge * (
        1.0 + kernel.noise_scale * observation_noise[data.indices]
    )
    diagonal_ids = np.arange(neighbors)
    matrix[:, diagonal_ids, diagonal_ids] += diagonal
    augmented = np.zeros((count, neighbors + 1, neighbors + 1), dtype=np.float64)
    augmented[:, :neighbors, :neighbors] = matrix
    augmented[:, :neighbors, neighbors] = 1.0
    augmented[:, neighbors, :neighbors] = 1.0
    rhs = np.zeros((count, neighbors + 1), dtype=np.float64)
    rhs[:, :neighbors] = query
    rhs[:, neighbors] = 1.0
    try:
        solution = np.linalg.solve(augmented, rhs[..., None]).squeeze(2)
    except np.linalg.LinAlgError:
        solution = np.stack(
            [np.linalg.lstsq(a, b, rcond=1e-7)[0] for a, b in zip(augmented, rhs)]
        )
    weight = solution[:, :neighbors]
    if kernel.positive_weights:
        weight = np.maximum(weight, 0.0)
        empty = weight.sum(axis=1) <= 1e-20
        if np.any(empty):
            weight[empty] = query[empty]
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-20)
    return weight.astype(np.float32)


def select_kernels(
    data: Neighborhood,
    bank_sh: np.ndarray,
    bank_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
    observation_noise: np.ndarray,
) -> tuple[Kernel, Kernel, list[dict[str, object]]]:
    rows = []
    for kernel in kernels():
        weight = kriging_weights(data, kernel, observation_noise)
        pred_sh = weighted_bank(bank_sh, data.indices, weight)
        pred_d = weighted_bank(bank_d, data.indices, weight)
        value = metrics(pred_sh, pred_d, truth_sh, truth_d)
        rows.append({"kernel": kernel.name, **value})
    sh_name = max(rows, key=lambda item: float(item["sh"]))["kernel"]
    d_name = max(rows, key=lambda item: float(item["d"]))["kernel"]
    lookup = {kernel.name: kernel for kernel in kernels()}
    return lookup[str(sh_name)], lookup[str(d_name)], rows


def predict_two(
    data: Neighborhood,
    sh_kernel: Kernel,
    d_kernel: Kernel,
    bank_sh: np.ndarray,
    bank_d: np.ndarray,
    observation_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    sh_weight = kriging_weights(data, sh_kernel, observation_noise)
    d_weight = kriging_weights(data, d_kernel, observation_noise)
    return (
        weighted_bank(bank_sh, data.indices, sh_weight),
        weighted_bank(bank_d, data.indices, d_weight),
        {
            "sh_effective_neighbors": summarize(
                1.0 / np.maximum(np.sum(sh_weight**2, axis=1), 1e-20)
            ),
            "d_effective_neighbors": summarize(
                1.0 / np.maximum(np.sum(d_weight**2, axis=1), 1e-20)
            ),
            "sh_negative_fraction": float(np.mean(sh_weight < 0.0)),
            "d_negative_fraction": float(np.mean(d_weight < 0.0)),
        },
    )


def choose_alpha(base: np.ndarray, model: np.ndarray, truth: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        blended = normalize(base + float(alpha) * (model - base))
        rows.append(
            {"alpha": float(alpha), "cosine": float(np.mean(row_cos(blended, truth)))}
        )
    return float(max(rows, key=lambda item: item["cosine"])["alpha"]), rows


def main() -> None:
    for path in (
        TARGET_CACHE,
        RELIABILITY_CACHE,
        BARRIER_CACHE,
        PATH_CACHE,
        BASE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    path_cache = np.load(PATH_CACHE)
    if not INTERACTION_CACHE.is_file():
        v293.build_feature_cache(path_cache)
    interaction = np.load(INTERACTION_CACHE)
    targets = np.load(TARGET_CACHE)
    truth_sh = normalize(targets["sh"].astype(np.float32))
    truth_d = normalize(targets["d"].astype(np.float32))
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    reliability = np.load(RELIABILITY_CACHE)["raw_half_cos"].astype(np.float32)
    raw_noise = np.maximum(1.0 - reliability, 0.0)
    scale = max(float(np.median(raw_noise[valid_ids][raw_noise[valid_ids] > 0.0])), 1e-5)
    observation_noise = np.clip(raw_noise / scale, 0.0, 12.0).astype(np.float32)
    barrier = np.load(BARRIER_CACHE)
    train_signature = barrier["train_signature"].astype(bool)
    test_signature = barrier["test_signature"].astype(bool)
    train_path = interaction["train_summary"].astype(np.float32)[:, 3:]
    test_path = interaction["test_summary"].astype(np.float32)[:, 3:]
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)

    outer_rows: list[dict[str, object]] = []
    oof_ids_parts: list[np.ndarray] = []
    oof_anchor_sh_parts: list[np.ndarray] = []
    oof_anchor_d_parts: list[np.ndarray] = []
    oof_model_sh_parts: list[np.ndarray] = []
    oof_model_d_parts: list[np.ndarray] = []
    selected_sh_kernels: list[str] = []
    selected_d_kernels: list[str] = []
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
        inner_data = neighborhood(
            inner_ids,
            inner_fit,
            train_pos,
            train_pos,
            train_signature,
            train_signature,
            train_path,
            train_path,
            False,
        )
        sh_kernel, d_kernel, inner_search = select_kernels(
            inner_data,
            truth_sh,
            truth_d,
            truth_sh[inner_ids],
            truth_d[inner_ids],
            observation_noise,
        )
        selected_sh_kernels.append(sh_kernel.name)
        selected_d_kernels.append(d_kernel.name)
        outer_data = neighborhood(
            outer_ids,
            outer_fit,
            train_pos,
            train_pos,
            train_signature,
            train_signature,
            train_path,
            train_path,
            False,
        )
        model_sh, model_d, weight_diag = predict_two(
            outer_data,
            sh_kernel,
            d_kernel,
            truth_sh,
            truth_d,
            observation_noise,
        )
        anchor_sh = knn_prediction(
            outer_ids, outer_fit, train_pos, train_pos, truth_sh, False
        )
        anchor_d = knn_prediction(
            outer_ids, outer_fit, train_pos, train_pos, truth_d, False
        )
        baseline = metrics(
            anchor_sh, anchor_d, truth_sh[outer_ids], truth_d[outer_ids]
        )
        model_metric = metrics(
            model_sh, model_d, truth_sh[outer_ids], truth_d[outer_ids]
        )
        outer_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "sh_kernel": sh_kernel.name,
                "d_kernel": d_kernel.name,
                "inner_search": inner_search,
                "baseline": baseline,
                "model": model_metric,
                "raw_delta": {
                    key: model_metric[key] - baseline[key]
                    for key in ("sh", "d", "weighted")
                },
                "weights": weight_diag,
            }
        )
        oof_ids_parts.append(outer_ids)
        oof_anchor_sh_parts.append(anchor_sh)
        oof_anchor_d_parts.append(anchor_d)
        oof_model_sh_parts.append(model_sh)
        oof_model_d_parts.append(model_d)
        print(
            f"v294 outer {fold + 1}/5 sh={sh_kernel.name} d={d_kernel.name} "
            f"delta={model_metric['weighted'] - baseline['weighted']:+.6f}",
            flush=True,
        )

    oof_ids = np.concatenate(oof_ids_parts)
    oof_anchor_sh = np.concatenate(oof_anchor_sh_parts)
    oof_anchor_d = np.concatenate(oof_anchor_d_parts)
    oof_model_sh = np.concatenate(oof_model_sh_parts)
    oof_model_d = np.concatenate(oof_model_d_parts)
    alpha_sh, alpha_sh_curve = choose_alpha(
        oof_anchor_sh, oof_model_sh, truth_sh[oof_ids]
    )
    alpha_d, alpha_d_curve = choose_alpha(
        oof_anchor_d, oof_model_d, truth_d[oof_ids]
    )
    fold_delta = []
    for row, ids, anchor_sh, anchor_d, model_sh, model_d in zip(
        outer_rows,
        oof_ids_parts,
        oof_anchor_sh_parts,
        oof_anchor_d_parts,
        oof_model_sh_parts,
        oof_model_d_parts,
    ):
        blended_sh = normalize(anchor_sh + alpha_sh * (model_sh - anchor_sh))
        blended_d = normalize(anchor_d + alpha_d * (model_d - anchor_d))
        blended = metrics(blended_sh, blended_d, truth_sh[ids], truth_d[ids])
        delta = blended["weighted"] - float(row["baseline"]["weighted"])
        row["blended"] = blended
        row["blended_delta"] = delta
        fold_delta.append(delta)
    mean_delta = float(np.mean(fold_delta))
    wins = int(np.sum(np.asarray(fold_delta) > 0.0))
    selected = bool(
        mean_delta >= MIN_OFFICIAL_DELTA
        and wins >= 3
        and (alpha_sh > 0.0 or alpha_d > 0.0)
    )

    final_kernels = None
    projection = None
    local = None
    if selected:
        final_inner_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], INNER_COUNT, 294_999
            ),
            valid_ids,
        )
        final_fit = np.setdiff1d(valid_ids, final_inner_ids)
        final_inner_data = neighborhood(
            final_inner_ids,
            final_fit,
            train_pos,
            train_pos,
            train_signature,
            train_signature,
            train_path,
            train_path,
            False,
        )
        final_sh_kernel, final_d_kernel, final_search = select_kernels(
            final_inner_data,
            truth_sh,
            truth_d,
            truth_sh[final_inner_ids],
            truth_d[final_inner_ids],
            observation_noise,
        )
        final_kernels = {
            "sh": final_sh_kernel.name,
            "d": final_d_kernel.name,
            "search": final_search,
        }
        test_data = neighborhood(
            np.arange(len(test_pos)),
            valid_ids,
            test_pos,
            train_pos,
            test_signature,
            train_signature,
            test_path,
            train_path,
            False,
        )
        test_model_sh, test_model_d, test_weight_diag = predict_two(
            test_data,
            final_sh_kernel,
            final_d_kernel,
            truth_sh,
            truth_d,
            observation_noise,
        )
        test_anchor_sh = knn_prediction(
            np.arange(len(test_pos)), valid_ids, test_pos, train_pos, truth_sh, False
        )
        test_anchor_d = knn_prediction(
            np.arange(len(test_pos)), valid_ids, test_pos, train_pos, truth_d, False
        )
        current_sh, current_d = v274.extract_current_marginals(BASE)
        target_sh = normalize(
            current_sh + alpha_sh * (test_model_sh - test_anchor_sh)
        )
        target_d = normalize(current_d + alpha_d * (test_model_d - test_anchor_d))
        np.savez_compressed(
            TARGETS_OUT,
            anchor_sh=test_anchor_sh,
            anchor_d=test_anchor_d,
            model_sh=test_model_sh,
            model_d=test_model_d,
            current_sh=current_sh,
            current_d=current_d,
            target_sh=target_sh,
            target_d=target_d,
            alpha_sh=np.asarray(alpha_sh, dtype=np.float32),
            alpha_d=np.asarray(alpha_d, dtype=np.float32),
        )
        projection_raw = v247.project(target_sh, target_d, OUTPUT)
        projection = {
            "weights": test_weight_diag,
            **{key: summarize(value) for key, value in projection_raw.items()},
        }
        local = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A heteroscedastic Matern Gaussian process can recover the clean planar "
            "spectral field better than equal-quality inverse-distance KNN by using "
            "official noise reliability and point-cloud propagation-state covariance."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP and even/odd antenna reliability",
                "official train/test planar coordinates",
                "official point-cloud blocker signatures",
                "v291 geometry-only interaction-depth summaries",
                "strict v239 carrier after official selection",
            ],
            "does_not_use": [
                "synthetic/Sionna channel labels or path coefficients",
                "v212/v217 teacher output",
                "hidden labels or local truth for kernel/alpha selection",
            ],
            "local_truth_usage": "read once after official selection" if selected else "not read",
        },
        "parameters": {
            "neighbors": NEIGHBORS,
            "baseline_k": BASELINE_K,
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "kernel_count": len(kernels()),
            "alpha_grid": ALPHA_GRID.tolist(),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
        },
        "noise": {
            "raw_half_cos": summarize(reliability[valid_ids]),
            "normalized_observation_noise": summarize(observation_noise[valid_ids]),
        },
        "outer_folds": outer_rows,
        "official_selection": {
            "alpha_sh": alpha_sh,
            "alpha_d": alpha_d,
            "alpha_sh_curve": alpha_sh_curve,
            "alpha_d_curve": alpha_d_curve,
            "fold_deltas": fold_delta,
            "mean_weighted_delta": mean_delta,
            "wins": wins,
            "selected_sh_kernels": selected_sh_kernels,
            "selected_d_kernels": selected_d_kernels,
            "selected": selected,
        },
        "final_kernels": final_kernels,
        "projection": projection,
        "local_gate": local,
        "outputs": {
            "targets": PATHS.relative(TARGETS_OUT) if selected else None,
            "prediction": PATHS.relative(OUTPUT) if selected else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
