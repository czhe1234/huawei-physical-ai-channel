#!/usr/bin/env python3
"""Generate v131: selective component-family boost.

v130 proved that family-KNN residual is too blunt when it controls the whole
support target.  v131 keeps the useful part: for each geometry/visibility
family, learn which individual AD components are repeatedly missing in train
bad points, then apply only those component-level positive boosts.

This is component-first rather than point-first:

* train truth/base residual builds family-specific missing-component maps;
* v130 provides a candidate direction, but only stable component bins survive;
* output changes only collapsed H/V/delay support and preserves v129 total
  power, phase, and pol/UE allocation.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import MH, MV, S_SC, summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from generate_v112_kernel_moe_path_support import normalize_support, row_cos
from generate_v128_badpoint_support_residual_templates import build_bad_mask, collapsed_support_from_ad
from generate_v130_family_knn_badpoint_residual import family_labels, scene_columns
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def build_family_component_maps(
    truth: np.ndarray,
    train_base: np.ndarray,
    bad_mask: np.ndarray,
    fam_train: np.ndarray,
    floor: float,
    min_count: int,
    ratio_clip: tuple[float, float],
) -> tuple[np.ndarray, dict[str, object]]:
    log_ratio = np.log(np.maximum(truth, floor)) - np.log(np.maximum(train_base, floor))
    pos = np.clip(log_ratio, 0.0, math.log(ratio_clip[1])).astype(np.float32)
    is_missing = (log_ratio > math.log(1.18)).astype(np.float32)
    power_weight = np.sqrt(np.maximum(truth, floor)).astype(np.float32)
    score_base = pos * (0.25 + 0.75 * power_weight / np.maximum(power_weight.reshape(power_weight.shape[0], -1).max(axis=1)[:, None, None, None], floor))

    maps = np.zeros((8, MH, MV, S_SC), dtype=np.float32)
    counts: dict[str, int] = {}
    global_idx = np.flatnonzero(bad_mask)
    for fam in range(8):
        idx = np.flatnonzero(bad_mask & (fam_train == fam))
        counts[str(fam)] = int(len(idx))
        if len(idx) < int(min_count):
            idx = global_idx
        mean_score = score_base[idx].mean(axis=0)
        freq = is_missing[idx].mean(axis=0)
        m = mean_score * np.sqrt(np.maximum(freq, 0.0))
        # Keep a compact high-confidence map.  This prevents broad family
        # priors from acting like v130's all-support residual.
        flat = m.reshape(-1)
        q = float(np.quantile(flat, 0.965))
        m = np.where(m >= q, m, 0.0).astype(np.float32)
        scale = float(np.quantile(m[m > 0], 0.95)) if np.any(m > 0) else 1.0
        maps[fam] = np.clip(m / max(scale, 1e-12), 0.0, 1.0).astype(np.float32)
    report = {
        "family_bad_counts": counts,
        "map_nonzero": {str(i): int(np.count_nonzero(maps[i])) for i in range(8)},
        "map_mean": {str(i): float(maps[i].mean()) for i in range(8)},
        "map_max": {str(i): float(maps[i].max()) for i in range(8)},
    }
    return maps, report


def synthetic_weights(conf: np.ndarray, count: int) -> np.ndarray:
    # Projection() only uses these as a confidence summary.  Build a distribution
    # whose max/entropy increases smoothly with component-family agreement.
    conf = np.clip(conf.astype(np.float32), 0.0, 1.0)
    w = np.empty((len(conf), count), dtype=np.float32)
    for i, c in enumerate(conf):
        top = 0.18 + 0.62 * float(c)
        rest = max(1.0 - top, 1e-6)
        w[i, 0] = top
        w[i, 1:] = rest / max(count - 1, 1)
    w /= np.maximum(w.sum(axis=1, keepdims=True), TINY)
    return w.astype(np.float32)


def selective_target(
    current: np.ndarray,
    direction: np.ndarray,
    fam_test: np.ndarray,
    maps: np.ndarray,
    regimes: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    log_dir = np.log(np.maximum(direction, args.floor)) - np.log(np.maximum(current, args.floor))
    pos_dir = np.clip(log_dir, 0.0, math.log(float(args.max_boost_ratio))).astype(np.float32)
    fam_maps = maps[fam_test].astype(np.float32)
    comp_score = pos_dir * fam_maps
    flat = comp_score.reshape(len(current), -1)
    topk = min(int(args.component_topk), flat.shape[1])
    ids = np.argpartition(flat, -topk, axis=1)[:, -topk:]
    mask = np.zeros_like(flat, dtype=np.float32)
    vals = np.take_along_axis(flat, ids, axis=1)
    # Keep only bins with actual positive direction and stable family prior.
    keep = vals > float(args.component_threshold)
    rows = np.arange(len(current))[:, None]
    mask[rows, ids] = keep.astype(np.float32)
    mask = mask.reshape(current.shape)

    selected_score = (comp_score * mask).reshape(len(current), -1)
    selected_mass = selected_score.sum(axis=1).astype(np.float32)
    selected_max = selected_score.max(axis=1).astype(np.float32)
    selected_count = mask.reshape(len(current), -1).sum(axis=1).astype(np.float32)
    conf = np.clip(np.log1p(selected_mass * 120.0) / float(args.conf_norm), 0.0, 1.0)
    conf *= np.clip(selected_count / max(float(args.component_topk) * 0.35, 1.0), 0.0, 1.0)

    regime_scale = np.array(
        [
            {
                "clear": float(args.clear_scale),
                "light": float(args.light_scale),
                "medium": float(args.medium_scale),
                "heavy": float(args.heavy_scale),
            }[str(r)]
            for r in regimes
        ],
        dtype=np.float32,
    )
    strength = np.clip(conf * regime_scale, 0.0, 1.0).astype(np.float32)
    lr = pos_dir * mask * strength[:, None, None, None]
    lr = np.clip(lr, 0.0, math.log(float(args.max_effective_ratio))).astype(np.float32)
    target = current.astype(np.float32) * np.exp(lr).astype(np.float32)
    target = np.maximum(target, float(args.floor)).astype(np.float32)
    target /= np.maximum(target.sum(axis=(1, 2, 3), keepdims=True), TINY)
    report = {
        "selected_mass": summarize(selected_mass),
        "selected_max": summarize(selected_max),
        "selected_count": summarize(selected_count),
        "component_conf": summarize(conf),
        "strength": summarize(strength),
    }
    return target.astype(np.float32), conf.astype(np.float32), report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v129/v129_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v131/v131_Round1_Test_Channel.npy")
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-base-ad", default=ASSETS + "/train_v125eq_crossfit_base_ad.npy")
    ap.add_argument("--train-base-support-out", default=ASSETS + "/cache_v128/train_v125eq_collapsed_support.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v129/v129_apex_badpoint_support_residual_target.npy")
    ap.add_argument("--direction-target", default=ASSETS + "/cache_v130/v130_family_knn_badpoint_residual_target.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v131/v131_selective_component_family_boost_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v131/v131_selective_component_family_boost_weights.npy")
    ap.add_argument("--maps-out", default=ASSETS + "/cache_v131/v131_family_component_maps.npy")
    ap.add_argument("--report", default=ASSETS + "/v131_selective_component_family_boost_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v131_selective_component_family_boost_candidate_pending.zip"))
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--ad-batch", type=int, default=10)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--bad-quantile", type=float, default=0.50)
    ap.add_argument("--min-bad-cos", type=float, default=0.978)
    ap.add_argument("--min-family-count", type=int, default=48)
    ap.add_argument("--component-topk", type=int, default=768)
    ap.add_argument("--component-threshold", type=float, default=0.000012)
    ap.add_argument("--max-boost-ratio", type=float, default=7.5)
    ap.add_argument("--max-effective-ratio", type=float, default=3.2)
    ap.add_argument("--conf-norm", type=float, default=4.2)
    ap.add_argument("--clear-scale", type=float, default=0.02)
    ap.add_argument("--light-scale", type=float, default=0.20)
    ap.add_argument("--medium-scale", type=float, default=0.82)
    ap.add_argument("--heavy-scale", type=float, default=1.05)
    ap.add_argument("--weight-count", type=int, default=12)
    ap.add_argument("--eta", default="clear:0.0002,light:0.020,medium:0.900,heavy:1.180")
    ap.add_argument("--min-gate", type=float, default=0.22)
    ap.add_argument("--max-gate", type=float, default=3.15)
    ap.add_argument("--conf-center", type=float, default=0.16)
    ap.add_argument("--conf-width", type=float, default=0.20)
    ap.add_argument("--entropy-low", type=float, default=0.18)
    ap.add_argument("--entropy-high", type=float, default=0.94)
    ap.add_argument("--gain-clip", default="0.42,3.80")
    ap.add_argument("--top-gain-clip", default="0.50,3.20")
    ap.add_argument("--top-components", type=int, default=3072)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    args = ap.parse_args()
    args.target_floor = float(args.floor)

    data = Path(args.data)
    pos_train = np.load(data / "Round1_Train_Pos.npy")
    pos_test = np.load(data / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    scene = np.load(args.scene_cache, allow_pickle=True)

    truth = normalize_support(np.load(args.truth_support, mmap_mode="r"), float(args.floor))
    train_base = collapsed_support_from_ad(args.train_base_ad, args.train_base_support_out, float(args.floor), int(args.ad_batch))
    current = normalize_support(np.load(args.current_target), float(args.floor))
    direction = normalize_support(np.load(args.direction_target), float(args.floor))

    bad_mask, train_cos = build_bad_mask(
        train_base,
        truth,
        regimes_train,
        float(args.bad_quantile),
        float(args.min_bad_cos),
    )
    fam_train, _ = family_labels(blocked_train, scene_columns(scene, "train"))
    fam_test, _ = family_labels(blocked_test, scene_columns(scene, "test"))
    maps, map_report = build_family_component_maps(
        truth,
        train_base,
        bad_mask,
        fam_train,
        float(args.floor),
        int(args.min_family_count),
        (0.12, float(args.max_boost_ratio)),
    )
    target, conf, sel_report = selective_target(current, direction, fam_test, maps, regimes_test, args)
    weights = synthetic_weights(conf, int(args.weight_count))

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target.astype(np.float32))
    np.save(args.weights_out, weights.astype(np.float32))
    np.save(args.maps_out, maps.astype(np.float32))

    proj = projection(target, weights, regimes_test, pos_test, args)
    payload = {
        "method": "v131_selective_component_family_boost",
        "base": args.base,
        "operation": "component-first selective boost: family-specific stable missing-component maps filter v130 direction; support-only projection preserving v129 total power/phase/pol-UE allocation",
        "args": vars(args),
        "bad_train": {
            "count": int(bad_mask.sum()),
            "fraction": float(np.mean(bad_mask)),
            "train_base_vs_truth_cos": summarize(train_cos),
            "bad_cos": summarize(train_cos[bad_mask]),
            "bad_regime_counts": {r: int(np.sum(regimes_train[bad_mask] == r)) for r in REGIMES},
            "bad_family_counts": {str(i): int(np.sum(fam_train[bad_mask] == i)) for i in range(8)},
        },
        "test_family_counts": {str(i): int(np.sum(fam_test == i)) for i in range(8)},
        "component_maps": map_report,
        "selector": sel_report,
        "target_cos": {
            "target_vs_current": summarize(row_cos(target, current)),
            "target_vs_direction": summarize(row_cos(target, direction)),
        },
        "projection": proj,
        "outputs": {
            "submission_npy": args.output,
            "target": args.target_out,
            "weights": args.weights_out,
            "maps": args.maps_out,
        },
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    files = [
        (args.output, "Round1_Test_Channel.npy"),
        (__file__, "generate_v131_selective_component_family_boost.py"),
        (ASSETS + "/generate_v130_family_knn_badpoint_residual.py", "generate_v130_family_knn_badpoint_residual.py"),
        (ASSETS + "/generate_v128_badpoint_support_residual_templates.py", "generate_v128_badpoint_support_residual_templates.py"),
        (ASSETS + "/generate_v121_support_template_manifold.py", "generate_v121_support_template_manifold.py"),
        (ASSETS + "/generate_v115_pointcloud_scene_support.py", "generate_v115_pointcloud_scene_support.py"),
        (ASSETS + "/generate_v112_kernel_moe_path_support.py", "generate_v112_kernel_moe_path_support.py"),
        (ASSETS + "/generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
        (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
        (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
        (PATHS.code_root / "project_paths.py", "project_paths.py"),
        (ASSETS + "/build_official_pointcloud_scene.py", "build_official_pointcloud_scene.py"),
        (ASSETS + "/build_v125_equivalent_crossfit_base.py", "build_v125_equivalent_crossfit_base.py"),
    ]
    Path(args.zip_out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_STORED) as zf:
        for src, arc in files:
            zf.write(src, arc)
    print(f"wrote {args.zip_out}", flush=True)


if __name__ == "__main__":
    main()
