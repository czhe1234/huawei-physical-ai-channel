#!/usr/bin/env python3
"""v265: strict official wall-scatterer delay-angle representation audit."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.spatial import cKDTree

import analyze_v254_route_topology_spectral_field as v254
import analyze_v264_patch_scatterer_transformer as v264
from generate_v87_easy_field_projection import MH, S_SC, summarize
from project_paths import PATHS


VERSION_NUMBER = 265
VERSION = "v265_scatterer_delay_angle_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
NNLS_CACHE = PATHS.cache_dir(264) / "v264_train_scatterer_nnls.npz"

VAL_COUNT = 320
VAL_SEED = 254
BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
ORIENTATIONS = tuple(float(value) for value in range(-120, 121, 15))
H_MODES = ("sin", "cos")
H_SIGNS = (-1.0, 1.0)
H_SHIFTS = (0.0, MH / 2.0)
S_SIGNS = (-1.0, 1.0)
FULL_TOP = 6
FIT_PROBE_COUNT = 320


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 1e-12)
    out /= np.maximum(out.sum(axis=tuple(range(1, out.ndim)), keepdims=True), 1e-30)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def circular_delta(values: np.ndarray, centers: np.ndarray, size: int) -> np.ndarray:
    delta = np.abs(values - centers[..., None])
    return np.minimum(delta, float(size) - delta)


def collapse_path_weights(weights: np.ndarray, delay: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = delay[:, :, None] + v264.SHIFTS[None, None, :]
    offset_weight = weights[:, : v264.PATHS_PER_POINT * len(v264.SHIFTS)].reshape(
        len(weights), v264.PATHS_PER_POINT, len(v264.SHIFTS)
    )
    path_weight = offset_weight.sum(axis=2)
    effective_delay = np.sum(offset_weight * shifted, axis=2) / np.maximum(path_weight, 1e-12)
    effective_delay = np.where(path_weight > 1e-8, effective_delay, delay)
    return path_weight.astype(np.float32), effective_delay.astype(np.float32)


def departure_azimuth(
    train_pos: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(train_pos)
    point = np.zeros((count, v264.PATHS_PER_POINT, 3), dtype=np.float64)
    point[:, 0] = train_pos
    point[:, 1:] = candidates["reflection_point"][:, : v264.REFLECTIONS]
    valid = np.ones((count, v264.PATHS_PER_POINT), dtype=bool)
    valid[:, 1:] = candidates["score"][:, : v264.REFLECTIONS] > 0.0
    patch_id = np.zeros((count, v264.PATHS_PER_POINT), dtype=np.int64)
    patch_id[:, 1:] = np.where(
        valid[:, 1:], candidates["patch_id"][:, : v264.REFLECTIONS].astype(np.int64) + 1, 0
    )
    vector = point - BS[None, None, :]
    azimuth = np.arctan2(vector[:, :, 1], vector[:, :, 0]).astype(np.float32)
    return azimuth, valid, patch_id


def h_center_for(
    azimuth: np.ndarray,
    orientation: float,
    h_mode: str,
    h_sign: float,
    h_shift: float,
) -> np.ndarray:
    relative = azimuth - math.radians(orientation)
    direction = np.sin(relative) if h_mode == "sin" else np.cos(relative)
    return np.mod(h_sign * direction * (MH / 2.0) + h_shift, float(MH)).astype(np.float32)


def nearest_capture(
    target: np.ndarray,
    ids: np.ndarray,
    path_weight: np.ndarray,
    delay: np.ndarray,
    h_center: np.ndarray,
    s_sign: float,
    valid: np.ndarray,
) -> float:
    s_bin = np.mod(np.rint(s_sign * delay[ids]).astype(np.int64), S_SC)
    h_bin = np.mod(np.rint(h_center[ids]).astype(np.int64), MH)
    rows = np.arange(len(ids), dtype=np.int64)[:, None]
    sampled = target[ids][rows, s_bin, h_bin]
    weight = path_weight[ids] * valid[ids]
    return float(np.mean(np.sum(weight * sampled, axis=1) / np.maximum(weight.sum(axis=1), 1e-12)))


def render_sh(
    path_weight: np.ndarray,
    delay: np.ndarray,
    h_center: np.ndarray,
    s_sign: float,
    valid: np.ndarray,
    s_sigma: float,
    h_sigma: float,
) -> np.ndarray:
    s_values = np.arange(S_SC, dtype=np.float32)
    h_values = np.arange(MH, dtype=np.float32)
    s_center = np.mod(s_sign * delay, float(S_SC))
    s_delta = circular_delta(s_values, s_center, S_SC)
    h_delta = circular_delta(h_values, np.mod(h_center, float(MH)), MH)
    s_kernel = np.exp(-0.5 * (s_delta / s_sigma) ** 2).astype(np.float32)
    h_kernel = np.exp(-0.5 * (h_delta / h_sigma) ** 2).astype(np.float32)
    weight = path_weight * valid
    out = np.einsum("bp,bps,bph->bsh", weight, s_kernel, h_kernel, optimize=True)
    return normalize(out)


def estimate_patch_offsets(
    target: np.ndarray,
    fit_ids: np.ndarray,
    path_weight: np.ndarray,
    delay: np.ndarray,
    h_center: np.ndarray,
    s_sign: float,
    valid: np.ndarray,
    patch_id: np.ndarray,
    patch_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    vector_sum = np.zeros(patch_count + 1, dtype=np.complex128)
    evidence = np.zeros(patch_count + 1, dtype=np.float64)
    occurrences = np.zeros(patch_count + 1, dtype=np.int64)
    local_delta = np.arange(-4, 5, dtype=np.int64)
    for point_id in fit_ids:
        for path_index in range(1, v264.PATHS_PER_POINT):
            pid = int(patch_id[point_id, path_index])
            mass = float(path_weight[point_id, path_index])
            if pid <= 0 or not valid[point_id, path_index] or mass < 2e-4:
                continue
            s_bin = int(np.rint(s_sign * delay[point_id, path_index])) % S_SC
            base = int(np.rint(h_center[point_id, path_index])) % MH
            bins = np.mod(base + local_delta, MH)
            profile = (
                target[point_id, (s_bin - 1) % S_SC, bins]
                + 2.0 * target[point_id, s_bin, bins]
                + target[point_id, (s_bin + 1) % S_SC, bins]
            )
            best = int(np.argmax(profile))
            delta = int(local_delta[best])
            strength = mass * float(profile[best])
            if strength <= 0.0:
                continue
            vector_sum[pid] += strength * np.exp(2j * np.pi * delta / MH)
            evidence[pid] += strength
            occurrences[pid] += 1
    offset = np.zeros(patch_count + 1, dtype=np.float32)
    known = evidence > 0.0
    raw = np.angle(vector_sum[known]) * MH / (2.0 * np.pi)
    shrink = evidence[known] / (evidence[known] + 0.0015)
    offset[known] = np.clip(raw * shrink, -4.0, 4.0).astype(np.float32)
    return offset, {
        "known_patch_count": int(np.sum(known[1:])),
        "offset": summarize(offset[known]),
        "evidence": summarize(evidence[known]),
        "occurrences": summarize(occurrences[known]),
    }


def apply_patch_offsets(h_center: np.ndarray, patch_id: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.mod(h_center + offset[np.clip(patch_id, 0, len(offset) - 1)], float(MH)).astype(np.float32)


def knn_anchor(
    fit_pos: np.ndarray,
    fit_target: np.ndarray,
    query_pos: np.ndarray,
) -> np.ndarray:
    distance, local = cKDTree(fit_pos).query(query_pos, k=16)
    weight = 1.0 / np.maximum(distance, 0.05) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return normalize(np.sum(weight[:, :, None, None] * fit_target[local], axis=1))


def main() -> None:
    for path in (TARGET_CACHE, TRAIN_CANDIDATES, NNLS_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    targets = np.load(TARGET_CACHE)
    target_sh = normalize(targets["sh"].astype(np.float32))
    target_d = normalize(targets["d"].astype(np.float32))
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    matched = v254.choose_seeded_matched_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, VAL_SEED
    )
    val_ids = np.intersect1d(matched, valid_ids)
    fit_ids = np.setdiff1d(valid_ids, val_ids)
    candidates = dict(np.load(TRAIN_CANDIDATES))
    nnls_data = np.load(NNLS_CACHE)
    nnls_weights = nnls_data["weights"].astype(np.float32)

    delay = np.zeros((len(train_pos), v264.PATHS_PER_POINT), dtype=np.float32)
    delay[:, 0] = np.linalg.norm(train_pos - BS[None, :], axis=1) / v264.LIGHT_SPEED * S_SC * v264.SUBCARRIER_SPACING
    delay[:, 1:] = (
        candidates["path_length"][:, : v264.REFLECTIONS]
        / v264.LIGHT_SPEED
        * S_SC
        * v264.SUBCARRIER_SPACING
    )
    path_weight, effective_delay = collapse_path_weights(nnls_weights, delay)
    azimuth, path_valid, patch_id = departure_azimuth(train_pos, candidates)

    ranked: list[dict[str, object]] = []
    for orientation in ORIENTATIONS:
        for h_mode in H_MODES:
            for h_sign in H_SIGNS:
                for h_shift in H_SHIFTS:
                    h_center = h_center_for(azimuth, orientation, h_mode, h_sign, h_shift)
                    for s_sign in S_SIGNS:
                        capture = nearest_capture(
                            target_sh,
                            fit_ids,
                            path_weight,
                            effective_delay,
                            h_center,
                            s_sign,
                            path_valid,
                        )
                        ranked.append(
                            {
                                "orientation": orientation,
                                "h_mode": h_mode,
                                "h_sign": h_sign,
                                "h_shift": h_shift,
                                "s_sign": s_sign,
                                "fit_capture": capture,
                            }
                        )
    ranked.sort(key=lambda item: float(item["fit_capture"]), reverse=True)

    probe_stride = max(len(fit_ids) // FIT_PROBE_COUNT, 1)
    fit_probe_ids = fit_ids[::probe_stride][:FIT_PROBE_COUNT]
    full_results: list[dict[str, object]] = []
    for spec in ranked[:FULL_TOP]:
        h_center = h_center_for(
            azimuth,
            float(spec["orientation"]),
            str(spec["h_mode"]),
            float(spec["h_sign"]),
            float(spec["h_shift"]),
        )
        for s_sigma in (0.75, 1.05):
            for h_sigma in (0.75, 1.10):
                fit_pred = render_sh(
                    path_weight[fit_probe_ids],
                    effective_delay[fit_probe_ids],
                    h_center[fit_probe_ids],
                    float(spec["s_sign"]),
                    path_valid[fit_probe_ids],
                    s_sigma,
                    h_sigma,
                )
                full_results.append(
                    {
                        **spec,
                        "s_sigma": s_sigma,
                        "h_sigma": h_sigma,
                        "fit_cos": float(np.mean(row_cos(fit_pred, target_sh[fit_probe_ids]))),
                    }
                )
    full_results.sort(key=lambda item: float(item["fit_cos"]), reverse=True)
    best = full_results[0]
    base_h_center = h_center_for(
        azimuth,
        float(best["orientation"]),
        str(best["h_mode"]),
        float(best["h_sign"]),
        float(best["h_shift"]),
    )
    val_path = render_sh(
        path_weight[val_ids],
        effective_delay[val_ids],
        base_h_center[val_ids],
        float(best["s_sign"]),
        path_valid[val_ids],
        float(best["s_sigma"]),
        float(best["h_sigma"]),
    )
    patch_offset, patch_report = estimate_patch_offsets(
        target_sh,
        fit_ids,
        path_weight,
        effective_delay,
        base_h_center,
        float(best["s_sign"]),
        path_valid,
        patch_id,
        int(np.max(patch_id)),
    )
    calibrated_h = apply_patch_offsets(base_h_center, patch_id, patch_offset)
    val_calibrated = render_sh(
        path_weight[val_ids],
        effective_delay[val_ids],
        calibrated_h[val_ids],
        float(best["s_sign"]),
        path_valid[val_ids],
        float(best["s_sigma"]),
        float(best["h_sigma"]),
    )
    anchor_sh = knn_anchor(train_pos[fit_ids, :2], target_sh[fit_ids], train_pos[val_ids, :2])
    anchor_d = v264.knn_anchor(train_pos[fit_ids, :2], target_d[fit_ids], train_pos[val_ids, :2])
    alpha_grid = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    blend_curve = []
    for alpha in alpha_grid:
        blended = normalize((1.0 - alpha) * anchor_sh + alpha * val_calibrated)
        blend_curve.append(float(np.mean(row_cos(blended, target_sh[val_ids]))))
    best_alpha_index = int(np.argmax(blend_curve))

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Stable official wall-scatterer delays and departure directions jointly explain "
            "the leaked delay-horizontal-angle support, including reusable patch-specific angle calibration."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H leaked S-H and PDP targets",
                "official train/test coordinates for matched holdout construction",
                "official point-cloud wall patches and analytic reflection points",
                "v264 official-train NNLS weights as an analysis-only same-point oracle",
            ],
            "does_not_use": [
                "Sionna or synthetic labels",
                "teacher output",
                "local replay truth",
                "hidden labels",
                "raw RT H",
            ],
        },
        "parameters": {
            "fit_count": int(len(fit_ids)),
            "fit_probe_count": int(len(fit_probe_ids)),
            "validation_count": int(len(val_ids)),
            "orientation_grid": list(ORIENTATIONS),
            "full_candidates": FULL_TOP,
        },
        "selection": {
            "top_capture": ranked[:20],
            "top_full_fit": full_results[:20],
            "best": best,
        },
        "validation": {
            "knn_sh_cos": summarize(row_cos(anchor_sh, target_sh[val_ids])),
            "path_sh_cos": summarize(row_cos(val_path, target_sh[val_ids])),
            "patch_calibrated_sh_cos": summarize(row_cos(val_calibrated, target_sh[val_ids])),
            "knn_d_cos": summarize(row_cos(anchor_d, target_d[val_ids])),
            "same_point_nnls_d_cos": summarize(
                row_cos(nnls_data["reconstruction"][val_ids], target_d[val_ids])
            ),
            "best_blend_alpha": float(alpha_grid[best_alpha_index]),
            "best_blend_cos": float(blend_curve[best_alpha_index]),
            "blend_gain": float(blend_curve[best_alpha_index] - blend_curve[0]),
            "blend_curve": [
                {"alpha": float(alpha), "cos": score}
                for alpha, score in zip(alpha_grid, blend_curve)
            ],
        },
        "patch_calibration": patch_report,
        "outputs": {"report": PATHS.relative(REPORT), "prediction": None, "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
