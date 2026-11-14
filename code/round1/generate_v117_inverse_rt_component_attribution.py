#!/usr/bin/env python3
"""Generate v117: official point-cloud inverse-RT component attribution.

v116 used official point-cloud path-family similarity directly at the support
level and dropped online.  The missing step is path/component identity: a wall
or LOS candidate is useful only if we know which angle-delay component family it
explains.

This version first performs an inverse-RT attribution on the training set:

1. Convert train truth support to collapsed AD power P(h, v, delay).
2. Generate official point-cloud LOS/reflection path candidates for every train
   and test point.
3. Match train top AD components to train path candidates by delay and BS AoD.
4. Store residual component offsets relative to the matched geometric path.
5. For every test path candidate, retrieve similar attributed train path samples
   and transfer their residual component offsets to build a test support target.

It still follows the successful v102+ submission rule: keep the base complex
phase/scalar and per-point total power, and only nudge the collapsed joint AD
support.  It uses only official data-derived geometry, not Sionna mesh/object/
material labels or precomputed Sionna path-token files.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from generate_v87_easy_field_projection import BS_POS, MH, MV, S_SC, summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from generate_v112_kernel_moe_path_support import (
    log_blend,
    normalize_support,
    row_cos,
    standardize,
)
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def circular_signed_delta(values: np.ndarray, centers: np.ndarray, period: int) -> np.ndarray:
    return ((values - centers + 0.5 * period) % period) - 0.5 * period


def circular_abs_delta(values: np.ndarray, centers: np.ndarray, period: int) -> np.ndarray:
    return np.abs(circular_signed_delta(values, centers, period))


def flat_to_hvs(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = ids // (MV * S_SC)
    rem = ids % (MV * S_SC)
    v = rem // S_SC
    s = rem % S_SC
    return h.astype(np.float32), v.astype(np.float32), s.astype(np.float32)


def top_components_from_support(support: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    flat = support.reshape(support.shape[0], -1)
    count = min(int(count), flat.shape[1])
    ids = np.argpartition(flat, -count, axis=1)[:, -count:]
    rows = np.arange(flat.shape[0])[:, None]
    order = np.argsort(flat[rows, ids], axis=1)[:, ::-1]
    ids = ids[rows, order]
    vals = flat[rows, ids]
    return ids.astype(np.int32), vals.astype(np.float32)


def path_angle_bins(
    tx_dir: np.ndarray,
    steering_scale: float,
    h_sign: int,
    v_sign: int,
    h_offset: float,
    v_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    h = (h_sign * steering_scale * tx_dir[..., 1] / (2.0 * np.pi) * MH + h_offset) % MH
    v = (v_sign * steering_scale * (-tx_dir[..., 2]) / (2.0 * np.pi) * MV + v_offset) % MV
    return h.astype(np.float32), v.astype(np.float32)


def make_official_paths(
    pos: np.ndarray,
    blocked: np.ndarray,
    refl_score: np.ndarray,
    refl_path: np.ndarray,
    refl_point: np.ndarray,
    refl_patch: np.ndarray,
    wall_patches: np.ndarray,
    delay_slope: float,
    delay_intercept: float,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Create LOS + one-bounce official point-cloud path candidates."""
    pos = np.asarray(pos, dtype=np.float32)
    n = len(pos)
    k_ref = min(int(args.reflection_candidates), refl_score.shape[1])
    p_count = 1 + k_ref

    valid = np.zeros((n, p_count), dtype=bool)
    path_type = np.zeros((n, p_count), dtype=np.int8)
    patch_id = np.full((n, p_count), -1, dtype=np.int32)
    length = np.full((n, p_count), np.inf, dtype=np.float32)
    score = np.zeros((n, p_count), dtype=np.float32)
    point = np.zeros((n, p_count, 3), dtype=np.float32)
    tx_dir = np.zeros((n, p_count, 3), dtype=np.float32)
    rx_dir = np.zeros((n, p_count, 3), dtype=np.float32)
    normal = np.zeros((n, p_count, 3), dtype=np.float32)
    patch_feat = np.zeros((n, p_count, 8), dtype=np.float32)

    rel = pos - BS_POS[None, :].astype(np.float32)
    direct_len = np.linalg.norm(rel, axis=1).astype(np.float32)
    direct_dir = rel / np.maximum(direct_len[:, None], 1e-6)
    valid[:, 0] = True
    path_type[:, 0] = 0
    length[:, 0] = direct_len
    # Do not hard-drop blocked LOS: Sionna-like data may still contain strong
    # deterministic paths in partially blocked regions.  This score is a soft
    # prior for attribution/query confidence only.
    score[:, 0] = (0.38 + 0.62 / (1.0 + np.maximum(blocked, 0.0) / 35.0)).astype(np.float32)
    point[:, 0, :] = (BS_POS[None, :].astype(np.float32) + pos) * 0.5
    tx_dir[:, 0, :] = direct_dir
    rx_dir[:, 0, :] = direct_dir

    r_score = np.nan_to_num(refl_score[:, :k_ref].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    r_path = np.nan_to_num(refl_path[:, :k_ref].astype(np.float32), nan=np.inf, posinf=np.inf, neginf=np.inf)
    r_point = np.nan_to_num(refl_point[:, :k_ref].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r_patch = np.asarray(refl_patch[:, :k_ref], dtype=np.int32)
    safe_patch = np.clip(r_patch, 0, max(len(wall_patches) - 1, 0))
    patch = wall_patches[safe_patch].astype(np.float32)
    patch_ok = (r_patch >= 0) & (r_patch < len(wall_patches))
    cand_ok = patch_ok & np.isfinite(r_path) & (r_score > float(args.min_reflection_score))

    valid[:, 1:] = cand_ok
    path_type[:, 1:] = 1
    patch_id[:, 1:] = np.where(cand_ok, r_patch, -1)
    length[:, 1:] = r_path
    score[:, 1:] = np.where(cand_ok, r_score, 0.0)
    point[:, 1:, :] = r_point
    tx = r_point - BS_POS[None, None, :].astype(np.float32)
    tx = tx / np.maximum(np.linalg.norm(tx, axis=2, keepdims=True), 1e-6)
    rx = pos[:, None, :] - r_point
    rx = rx / np.maximum(np.linalg.norm(rx, axis=2, keepdims=True), 1e-6)
    tx_dir[:, 1:, :] = tx
    rx_dir[:, 1:, :] = rx
    n2 = patch[..., :2]
    n3 = np.concatenate([n2, np.zeros((*n2.shape[:2], 1), dtype=np.float32)], axis=2)
    n3 = n3 / np.maximum(np.linalg.norm(n3, axis=2, keepdims=True), 1e-6)
    normal[:, 1:, :] = n3

    patch_len = np.maximum(patch[..., 4] - patch[..., 3], 0.0)
    patch_h = np.maximum(patch[..., 6] - patch[..., 5], 0.0)
    patch_count = np.log1p(np.maximum(patch[..., 7], 0.0))
    patch_feat[:, 1:, :] = np.stack(
        [
            patch[..., 0],
            patch[..., 1],
            patch[..., 2] / 260.0,
            ((patch[..., 3] + patch[..., 4]) * 0.5) / 260.0,
            ((patch[..., 5] + patch[..., 6]) * 0.5) / 35.0,
            patch_len / 90.0,
            patch_h / 35.0,
            patch_count / 9.0,
        ],
        axis=2,
    )

    delay = (float(delay_slope) * length + float(delay_intercept) + float(args.delay_offset)).astype(np.float32)
    hbin, vbin = path_angle_bins(
        tx_dir,
        float(args.steering_scale),
        int(args.h_sign),
        int(args.v_sign),
        float(args.h_offset),
        float(args.v_offset),
    )
    return {
        "valid": valid,
        "path_type": path_type,
        "patch_id": patch_id,
        "length": length,
        "delay": delay,
        "hbin": hbin,
        "vbin": vbin,
        "score": score,
        "point": point,
        "tx_dir": tx_dir,
        "rx_dir": rx_dir,
        "normal": normal,
        "patch_feat": patch_feat,
    }


def path_feature_matrix(paths: dict[str, np.ndarray], pos: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    n, p = paths["valid"].shape
    rel = pos - BS_POS[None, :].astype(np.float32)
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    reg = regime_of(blocked)
    reg_oh = np.stack([(reg == r).astype(np.float32) for r in REGIMES], axis=1)
    reg_rep = np.repeat(reg_oh[:, None, :], p, axis=1)
    ptype = paths["path_type"].astype(np.float32)
    ptype_oh = np.stack([(ptype == 0).astype(np.float32), (ptype == 1).astype(np.float32)], axis=2)
    pid = np.maximum(paths["patch_id"].astype(np.float32), 0.0) + 1.0
    freqs = np.asarray([0.011, 0.023, 0.041, 0.071], dtype=np.float32)
    pid_hash = np.concatenate([np.sin(pid[..., None] * freqs), np.cos(pid[..., None] * freqs)], axis=2)
    pid_hash *= (paths["patch_id"] >= 0)[..., None]
    valid = paths["valid"].astype(np.float32)
    inc = np.abs(np.sum(paths["tx_dir"] * paths["normal"], axis=2, keepdims=True))
    spec = np.abs(
        np.sum(paths["tx_dir"] * paths["normal"], axis=2, keepdims=True)
        + np.sum(paths["rx_dir"] * paths["normal"], axis=2, keepdims=True)
    )
    base = np.stack(
        [
            np.repeat((pos[:, 0] - 120.0) / 120.0, p),
            np.repeat((pos[:, 1] + 80.0) / 120.0, p),
            np.repeat(dist / 250.0, p),
            np.repeat(np.sin(az), p),
            np.repeat(np.cos(az), p),
            valid.reshape(-1),
            paths["score"].reshape(-1),
            paths["length"].reshape(-1) / 260.0,
            paths["delay"].reshape(-1) / 24.0,
            paths["hbin"].reshape(-1) / MH,
            paths["vbin"].reshape(-1) / MV,
            np.repeat(blocked / 160.0, p),
        ],
        axis=1,
    ).reshape(n, p, -1)
    feat = np.concatenate(
        [
            base,
            ptype_oh,
            reg_rep,
            paths["tx_dir"],
            paths["rx_dir"],
            paths["normal"],
            inc,
            spec,
            paths["point"] / np.asarray([260.0, 260.0, 35.0], dtype=np.float32),
            paths["patch_feat"],
            pid_hash,
        ],
        axis=2,
    )
    return np.nan_to_num(feat.reshape(n * p, -1), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def build_attributed_samples(
    truth_support: np.ndarray,
    paths: dict[str, np.ndarray],
    path_features_std: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    comp_ids, comp_vals = top_components_from_support(truth_support, int(args.train_top_components))
    h, v, s = flat_to_hvs(comp_ids)
    total = np.maximum(truth_support.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
    comp_share = comp_vals / total[:, None]

    de = np.abs(s[:, :, None] - paths["delay"][:, None, :])
    he = circular_abs_delta(h[:, :, None], paths["hbin"][:, None, :], MH)
    ve = circular_abs_delta(v[:, :, None], paths["vbin"][:, None, :], MV)
    valid = paths["valid"][:, None, :] & np.isfinite(de) & np.isfinite(he) & np.isfinite(ve)
    geom_score = (de / float(args.assign_delay_weight)) ** 2
    geom_score += (he / float(args.assign_h_weight)) ** 2
    geom_score += (ve / float(args.assign_v_weight)) ** 2
    path_prior = np.maximum(paths["score"][:, None, :], 1e-4)
    geom_score -= float(args.path_score_bonus) * np.log(path_prior)
    geom_score = np.where(valid, geom_score, np.inf)
    best = np.argmin(geom_score, axis=2)
    row = np.arange(comp_ids.shape[0])[:, None]
    col = np.arange(comp_ids.shape[1])[None, :]
    best_score = geom_score[row, col, best].astype(np.float32)
    best_de = de[row, col, best].astype(np.float32)
    best_he = he[row, col, best].astype(np.float32)
    best_ve = ve[row, col, best].astype(np.float32)
    best_valid = paths["valid"][row, best]
    conf = np.exp(-0.5 * np.minimum(best_score, 60.0)).astype(np.float32)
    conf *= np.power(np.maximum(paths["score"][row, best], 0.0), float(args.path_score_power)).astype(np.float32)
    keep = (
        best_valid
        & (best_de <= float(args.keep_delay_err))
        & (best_he <= float(args.keep_h_err))
        & (best_ve <= float(args.keep_v_err))
        & (conf >= float(args.min_assignment_conf))
        & (comp_share >= float(args.min_component_share))
    )

    rr, cc = np.nonzero(keep)
    bp = best[rr, cc]
    p_count = paths["valid"].shape[1]
    flat_path = rr * p_count + bp
    dh = circular_signed_delta(h[rr, cc], paths["hbin"][rr, bp], MH).astype(np.float32)
    dv = circular_signed_delta(v[rr, cc], paths["vbin"][rr, bp], MV).astype(np.float32)
    ds = (s[rr, cc] - paths["delay"][rr, bp]).astype(np.float32)
    sample_weight = (comp_share[rr, cc] * conf[rr, cc]).astype(np.float32)
    sample_type = paths["path_type"][rr, bp].astype(np.int8)
    sample = {
        "point_index": rr.astype(np.int32),
        "component_rank": cc.astype(np.int16),
        "path_index": bp.astype(np.int16),
        "path_type": sample_type,
        "path_flat_index": flat_path.astype(np.int32),
        "features": path_features_std[flat_path].astype(np.float32),
        "dh": dh,
        "dv": dv,
        "ds": ds,
        "component_share": comp_share[rr, cc].astype(np.float32),
        "assignment_conf": conf[rr, cc].astype(np.float32),
        "sample_weight": sample_weight,
        "delay_err": best_de[rr, cc].astype(np.float32),
        "h_err": best_he[rr, cc].astype(np.float32),
        "v_err": best_ve[rr, cc].astype(np.float32),
        "component_ids": comp_ids[rr, cc].astype(np.int32),
    }
    assigned_power = float(np.sum(comp_share[keep]) / max(float(np.sum(comp_share)), TINY))
    summary = {
        "sample_count": int(len(rr)),
        "top_component_count": int(comp_ids.size),
        "kept_component_frac": float(np.mean(keep)),
        "assigned_top1_frac": float(np.mean(keep[:, 0])),
        "assigned_power_share_among_top": assigned_power,
        "sample_weight_sum": float(np.sum(sample_weight)),
        "delay_err_p50": float(np.percentile(best_de[np.isfinite(best_de)], 50)),
        "delay_err_p90": float(np.percentile(best_de[np.isfinite(best_de)], 90)),
        "h_err_p50": float(np.percentile(best_he[np.isfinite(best_he)], 50)),
        "h_err_p90": float(np.percentile(best_he[np.isfinite(best_he)], 90)),
        "v_err_p50": float(np.percentile(best_ve[np.isfinite(best_ve)], 50)),
        "v_err_p90": float(np.percentile(best_ve[np.isfinite(best_ve)], 90)),
        "kept_los_frac": float(np.mean(sample_type == 0)) if len(sample_type) else 0.0,
        "kept_reflection_frac": float(np.mean(sample_type == 1)) if len(sample_type) else 0.0,
    }
    return sample, summary


def build_inverse_rt_target(
    current: np.ndarray,
    test_paths: dict[str, np.ndarray],
    test_features_std: np.ndarray,
    sample: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    n, p_count = test_paths["valid"].shape
    raw = np.full((n, MH, MV, S_SC), float(args.target_floor), dtype=np.float32)
    contrib_weights: list[list[float]] = [[] for _ in range(n)]

    type_reports: dict[str, dict[str, float]] = {}
    for ptype in (0, 1):
        mask_s = sample["path_type"] == ptype
        if int(mask_s.sum()) < 20:
            continue
        feat_s = sample["features"][mask_s]
        tree = cKDTree(feat_s)
        sample_idx = np.flatnonzero(mask_s)
        qmask = (test_paths["path_type"].reshape(-1) == ptype) & test_paths["valid"].reshape(-1)
        if not np.any(qmask):
            continue
        qflat = np.flatnonzero(qmask)
        qfeat = test_features_std[qflat]
        k = min(int(args.path_knn), feat_s.shape[0])
        d, loc = tree.query(qfeat, k=k, workers=-1)
        if k == 1:
            d = d[:, None]
            loc = loc[:, None]
        w = 1.0 / np.maximum(d, float(args.path_knn_floor)) ** float(args.path_knn_power)
        # Component attribution confidence and component share are part of the
        # transferred target, not just the distance kernel.
        rows_global = sample_idx[loc]
        w *= sample["sample_weight"][rows_global]
        q_score = test_paths["score"].reshape(-1)[qflat]
        w *= np.maximum(q_score[:, None], 0.0) ** float(args.query_path_score_power)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), TINY)

        q_point = qflat // p_count
        q_path = qflat % p_count
        pred_h = np.rint(test_paths["hbin"][q_point, q_path, None] + sample["dh"][rows_global]).astype(np.int64) % MH
        pred_v = np.rint(test_paths["vbin"][q_point, q_path, None] + sample["dv"][rows_global]).astype(np.int64) % MV
        pred_s = np.rint(test_paths["delay"][q_point, q_path, None] + sample["ds"][rows_global]).astype(np.int64)
        pred_s = np.clip(pred_s, 0, S_SC - 1)
        for i in range(len(qflat)):
            qq = int(q_point[i])
            vals = w[i].astype(np.float32)
            np.add.at(raw[qq], (pred_h[i], pred_v[i], pred_s[i]), vals)
            contrib_weights[qq].extend(vals[: min(len(vals), int(args.report_weight_count))].tolist())
        type_reports[str(ptype)] = {
            "query_path_count": int(len(qflat)),
            "sample_count": int(mask_s.sum()),
            "distance_p50": float(np.percentile(d, 50)),
            "distance_p90": float(np.percentile(d, 90)),
            "weight_max_p50": float(np.percentile(w.max(axis=1), 50)),
            "weight_max_p90": float(np.percentile(w.max(axis=1), 90)),
        }

    target = raw / np.maximum(raw.sum(axis=(1, 2, 3), keepdims=True), TINY)
    # Lightly mix in current to prevent pure sparse mis-attribution from
    # deleting stable components before the final projection confidence gate.
    attr_mix = log_blend(
        [current, target],
        np.column_stack(
            [
                np.full(n, 1.0 - float(args.attribution_inner_mix), dtype=np.float32),
                np.full(n, float(args.attribution_inner_mix), dtype=np.float32),
            ]
        ),
        float(args.target_floor),
        int(args.batch),
    )

    max_len = max(max((len(x) for x in contrib_weights), default=1), 1)
    out_k = min(max_len, int(args.point_weight_count))
    point_w = np.zeros((n, out_k), dtype=np.float32)
    for i, vals in enumerate(contrib_weights):
        if not vals:
            point_w[i, 0] = 1.0
            continue
        arr = np.sort(np.asarray(vals, dtype=np.float32))[::-1][:out_k]
        point_w[i, : len(arr)] = arr
        point_w[i] /= max(float(point_w[i].sum()), TINY)
    report = {
        "path_query": type_reports,
        "raw_vs_current": summarize(row_cos(target, current)),
        "mixed_vs_current": summarize(row_cos(attr_mix, current)),
        "point_weight_max": summarize(point_w.max(axis=1)),
        "point_weight_entropy": summarize(
            (-(point_w * np.log(np.maximum(point_w, 1e-12))).sum(axis=1) / math.log(point_w.shape[1]))
        ),
    }
    return attr_mix.astype(np.float32), point_w.astype(np.float32), report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v115/v115_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v117/v117_Round1_Test_Channel.npy")
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v115/v115_pointcloud_scene_support_target.npy")
    ap.add_argument("--v112-anchor", default=ASSETS + "/cache_v112/v112_kernel_moe_path_support_target.npy")
    ap.add_argument("--v111-anchor", default=ASSETS + "/cache_v111/v111_manual_kernel_path_support_target.npy")
    ap.add_argument("--v109-anchor", default=ASSETS + "/cache_v109/v109_accelerated_momentum_path_support_target.npy")
    ap.add_argument("--delay-fit", default=ASSETS + "/delay_length_fit.npz")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v117/v117_inverse_rt_component_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v117/v117_inverse_rt_component_weights.npy")
    ap.add_argument("--attrib-out", default=ASSETS + "/cache_v117/v117_inverse_rt_attributed_samples.npz")
    ap.add_argument("--report", default=ASSETS + "/v117_inverse_rt_component_attribution_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v117_inverse_rt_component_attribution_candidate_pending.zip"))
    ap.add_argument("--candidate-zip", default=str(PATHS.round1_versions / "Round1_Submission_candidate_v117_inverse_rt_component_attribution.zip"))

    ap.add_argument("--reflection-candidates", type=int, default=8)
    ap.add_argument("--min-reflection-score", type=float, default=0.015)
    ap.add_argument("--train-top-components", type=int, default=112)
    ap.add_argument("--assign-delay-weight", type=float, default=2.0)
    ap.add_argument("--assign-h-weight", type=float, default=2.5)
    ap.add_argument("--assign-v-weight", type=float, default=1.5)
    ap.add_argument("--keep-delay-err", type=float, default=2.3)
    ap.add_argument("--keep-h-err", type=float, default=3.1)
    ap.add_argument("--keep-v-err", type=float, default=2.4)
    ap.add_argument("--path-score-bonus", type=float, default=0.28)
    ap.add_argument("--path-score-power", type=float, default=0.25)
    ap.add_argument("--min-assignment-conf", type=float, default=0.018)
    ap.add_argument("--min-component-share", type=float, default=1e-5)
    ap.add_argument("--path-knn", type=int, default=72)
    ap.add_argument("--path-knn-power", type=float, default=2.65)
    ap.add_argument("--path-knn-floor", type=float, default=0.38)
    ap.add_argument("--query-path-score-power", type=float, default=0.50)
    ap.add_argument("--attribution-inner-mix", type=float, default=0.74)
    ap.add_argument("--point-weight-count", type=int, default=96)
    ap.add_argument("--report-weight-count", type=int, default=128)

    ap.add_argument("--steering-scale", type=float, default=4.71238898038469)
    ap.add_argument("--h-sign", type=int, default=-1)
    ap.add_argument("--v-sign", type=int, default=1)
    ap.add_argument("--delay-offset", type=float, default=-1.0)
    ap.add_argument("--h-offset", type=float, default=-1.0)
    ap.add_argument("--v-offset", type=float, default=1.0)

    ap.add_argument("--eta", default="clear:0.0005,light:0.010,medium:0.760,heavy:0.940")
    ap.add_argument("--min-gate", type=float, default=0.20)
    ap.add_argument("--max-gate", type=float, default=2.85)
    ap.add_argument("--conf-center", type=float, default=0.040)
    ap.add_argument("--conf-width", type=float, default=0.080)
    ap.add_argument("--entropy-low", type=float, default=0.15)
    ap.add_argument("--entropy-high", type=float, default=0.92)
    ap.add_argument("--gain-clip", default="0.34,4.05")
    ap.add_argument("--top-gain-clip", default="0.50,2.95")
    ap.add_argument("--top-components", type=int, default=2560)
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    args = ap.parse_args()
    args.target_floor = float(args.floor)

    data = Path(args.data)
    pos_train = np.load(data / "Round1_Train_Pos.npy").astype(np.float32)
    pos_test = np.load(data / "Round1_Test_Pos.npy").astype(np.float32)
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float32)
    blocked_test = geo["test_blocked"].astype(np.float32)
    regimes_test = regime_of(blocked_test)

    scene = np.load(args.scene_cache, allow_pickle=True)
    wall_patches = scene["wall_patches"].astype(np.float32)
    delay_fit = np.load(args.delay_fit)
    delay_slope = float(delay_fit["slope"])
    delay_intercept = float(delay_fit["intercept"])

    print("load supports", flush=True)
    truth = np.load(args.truth_support, mmap_mode="r").astype(np.float32)
    truth_norm = normalize_support(truth, float(args.floor))
    current = normalize_support(np.load(args.current_target), float(args.floor))
    v112_anchor = normalize_support(np.load(args.v112_anchor), float(args.floor))
    v111_anchor = normalize_support(np.load(args.v111_anchor), float(args.floor))
    v109_anchor = normalize_support(np.load(args.v109_anchor), float(args.floor))

    print("build official LOS/reflection path candidates", flush=True)
    train_paths = make_official_paths(
        pos_train,
        blocked_train,
        scene["train_reflection_score"],
        scene["train_reflection_path_length"],
        scene["train_reflection_point"],
        scene["train_reflection_patch_id"],
        wall_patches,
        delay_slope,
        delay_intercept,
        args,
    )
    test_paths = make_official_paths(
        pos_test,
        blocked_test,
        scene["test_reflection_score"],
        scene["test_reflection_path_length"],
        scene["test_reflection_point"],
        scene["test_reflection_patch_id"],
        wall_patches,
        delay_slope,
        delay_intercept,
        args,
    )

    print("standardize path-candidate features", flush=True)
    train_feat = path_feature_matrix(train_paths, pos_train, blocked_train)
    test_feat = path_feature_matrix(test_paths, pos_test, blocked_test)
    train_feat_std, test_feat_std = standardize(train_feat, test_feat)

    print("attribute train components to official paths", flush=True)
    sample, attrib_summary = build_attributed_samples(truth_norm, train_paths, train_feat_std, args)
    Path(args.attrib_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.attrib_out, **sample)

    print("transfer attributed residual component offsets to test paths", flush=True)
    attr_target, point_w, transfer_report = build_inverse_rt_target(current, test_paths, test_feat_std, sample, args)

    # Final target keeps v115 as the fallback and adds inverse-RT attribution as
    # the main new expert.  Small anchors keep the target in the high-scoring
    # support family when attribution is uncertain.
    inv_conf = np.clip(point_w.max(axis=1), 0.0, 1.0)
    inv_agree = row_cos(attr_target, current).astype(np.float32)
    reg_strength = {"clear": 0.010, "light": 0.055, "medium": 0.720, "heavy": 0.880}
    strength = np.array([reg_strength[str(r)] for r in regimes_test], dtype=np.float32)
    strength *= (0.60 + 0.40 * np.clip((inv_conf - 0.025) / 0.10, 0.0, 1.0)).astype(np.float32)
    strength *= (0.72 + 0.28 * np.clip((inv_agree - 0.82) / 0.16, 0.0, 1.0)).astype(np.float32)
    strength = np.clip(strength, 0.0, 0.82).astype(np.float32)
    coeff = np.zeros((len(current), 5), dtype=np.float32)
    coeff[:, 0] = 1.0 - strength
    coeff[:, 1] = 0.900 * strength
    coeff[:, 2] = 0.060 * strength
    coeff[:, 3] = 0.028 * strength
    coeff[:, 4] = 0.012 * strength
    target = log_blend([current, attr_target, v112_anchor, v111_anchor, v109_anchor], coeff, float(args.floor), int(args.batch))

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target.astype(np.float32))
    np.save(args.weights_out, point_w.astype(np.float32))

    print("project support target onto v115 complex channel", flush=True)
    proj = projection(target, point_w, regimes_test, pos_test, args)

    payload = {
        "method": "v117_inverse_rt_component_attribution",
        "base": args.base,
        "operation": "official point-cloud inverse-RT: train AD components are softly assigned to LOS/reflection path candidates; test path candidates retrieve attributed train residual component offsets; only collapsed joint AD support is nudged; phase/scalar/total power preserved",
        "online_context": {"previous_best": "v115 0.497298", "v116_feedback": 0.496164},
        "uses": {
            "official_point_cloud_scene_cache": args.scene_cache,
            "sionna_path_candidates": False,
            "sionna_mesh_object_material_labels": False,
        },
        "args": vars(args),
        "attribution": attrib_summary,
        "transfer": transfer_report,
        "target_cos": {
            "attr_vs_current": summarize(row_cos(attr_target, current)),
            "target_vs_current": summarize(row_cos(target, current)),
            "target_vs_attr": summarize(row_cos(target, attr_target)),
            "target_vs_v112": summarize(row_cos(target, v112_anchor)),
            "target_vs_v111": summarize(row_cos(target, v111_anchor)),
            "target_vs_v109": summarize(row_cos(target, v109_anchor)),
            "inverse_conf": summarize(inv_conf),
            "inverse_agreement": summarize(inv_agree),
            "blend_strength": summarize(strength),
        },
        "projection": proj,
        "outputs": {
            "submission_npy": args.output,
            "target": args.target_out,
            "weights": args.weights_out,
            "attribution": args.attrib_out,
        },
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    def write_zip(path: str) -> None:
        files = [
            (args.output, "Round1_Test_Channel.npy"),
            (__file__, "generate_v117_inverse_rt_component_attribution.py"),
            (ASSETS + "/generate_v115_pointcloud_scene_support.py", "generate_v115_pointcloud_scene_support.py"),
            (ASSETS + "/generate_v112_kernel_moe_path_support.py", "generate_v112_kernel_moe_path_support.py"),
            (ASSETS + "/generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
            (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
            (ASSETS + "/build_official_pointcloud_scene.py", "build_official_pointcloud_scene.py"),
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            for src, arc in files:
                zf.write(src, arc)
        print(f"wrote {path}", flush=True)

    write_zip(args.zip_out)
    if args.candidate_zip:
        write_zip(args.candidate_zip)


if __name__ == "__main__":
    main()
