#!/usr/bin/env python3
"""Generate v119: hybrid metric-aligned support projection.

This version tests the next scoring/physics hypothesis after v115:

The online score may care about several related support objects at once:

* conventional collapsed joint AD support P(H,V,delay);
* per-polarization / per-UE block support shape;
* leaked platform PAS-like S-H-V structure.

v115 only nudges collapsed P(H,V,delay).  v119 keeps v115 as the complex
backbone, preserves phase/scalar/total power, and adds a mild full-AD block
gain learned by official point-cloud geometry kNN from train truth.

It uses only official data-derived point-cloud geometry and train channels.
No Sionna mesh/object/material labels are required.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MP,
    MV,
    N_ANT,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
    row_cos,
    summarize,
)
from generate_v112_kernel_moe_path_support import (
    entropy01,
    normalize_support,
    pairwise_knn,
    standardize,
    weights_from_dist,
)
from generate_v115_pointcloud_scene_support import official_scene_embedding
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def parse_regime_values(text: str, default: float) -> dict[str, float]:
    out = {r: float(default) for r in REGIMES}
    if not text:
        return out
    for item in text.split(","):
        if not item:
            continue
        key, value = item.split(":")
        key = key.strip()
        if key not in out:
            raise ValueError(f"unknown regime {key!r}")
        out[key] = float(value)
    return out


def log_blend(arrays: list[np.ndarray], coeff: np.ndarray, floor: float, batch: int) -> np.ndarray:
    n = arrays[0].shape[0]
    out = np.empty_like(arrays[0], dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        acc = np.zeros_like(arrays[0][start:end], dtype=np.float64)
        for j, arr in enumerate(arrays):
            c = coeff[start:end, j]
            acc += c.reshape((-1,) + (1,) * (arr.ndim - 1)) * np.log(np.maximum(arr[start:end], floor))
        val = np.exp(acc).astype(np.float32)
        val /= np.maximum(val.sum(axis=tuple(range(1, val.ndim)), keepdims=True), TINY)
        out[start:end] = val
    return out


def full_log_average(
    base_ad_path: str,
    residual_ad_path: str,
    neighbor_idx: np.ndarray,
    weights: np.ndarray,
    out_path: str,
    floor: float,
    batch: int,
) -> np.ndarray:
    """Log-average train truth full AD power into a test full-support target."""

    base_ad = np.load(base_ad_path, mmap_mode="r")
    residual_ad = np.load(residual_ad_path, mmap_mode="r")
    n = neighbor_idx.shape[0]
    shape = (n,) + tuple(base_ad.shape[1:])
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        ids = neighbor_idx[start:end]
        w = weights[start:end].astype(np.float64)
        b, k = ids.shape
        acc = np.zeros((b,) + tuple(base_ad.shape[1:]), dtype=np.float64)
        # Keep the implementation memory bounded by processing each neighbor
        # column separately.  This is slower but avoids materializing B*K full
        # AD tensors at once.
        for j in range(k):
            truth = np.asarray(base_ad[ids[:, j]], dtype=np.complex64)
            truth += np.asarray(residual_ad[ids[:, j]], dtype=np.complex64)
            p = (truth.real**2 + truth.imag**2).astype(np.float32)
            p /= np.maximum(p.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
            acc += w[:, j].reshape((-1, 1, 1, 1, 1, 1)) * np.log(np.maximum(p, floor))
        val = np.exp(acc).astype(np.float32)
        val /= np.maximum(val.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        out[start:end] = val
        out.flush()
        print(f"full target {end}/{n}", flush=True)
    return out


def project_hybrid(
    full_target: np.ndarray,
    coll_target: np.ndarray,
    weights: np.ndarray,
    regimes: np.ndarray,
    pos_test: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    base = np.load(args.base, mmap_mode="r")
    beta_s = beta_subcarriers(float(args.beta), float(args.gamma))
    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)

    eta_coll_base = parse_regime_values(args.eta_collapsed, 0.0)
    eta_full_base = parse_regime_values(args.eta_full, 0.0)
    gain_clip = tuple(float(x) for x in args.gain_clip.split(","))
    top_gain_clip = tuple(float(x) for x in args.top_gain_clip.split(","))

    maxw = weights.max(axis=1)
    ent = entropy01(weights)
    conf = np.clip((maxw - float(args.conf_center)) / max(float(args.conf_width), 1e-6), 0.0, 1.0)
    conf *= np.clip((float(args.entropy_high) - ent) / max(float(args.entropy_high) - float(args.entropy_low), 1e-6), 0.0, 1.0)
    gate = (float(args.min_gate) + (float(args.max_gate) - float(args.min_gate)) * conf).astype(np.float32)
    eta_coll = np.asarray([eta_coll_base[str(r)] for r in regimes], dtype=np.float32) * gate
    eta_full = np.asarray([eta_full_base[str(r)] for r in regimes], dtype=np.float32) * gate

    stats: dict[str, object] = {
        "eta_collapsed": eta_coll.tolist(),
        "eta_full": eta_full.tolist(),
        "knn_max": maxw.tolist(),
        "knn_entropy": ent.tolist(),
        "confidence": conf.tolist(),
        "power_ratio": [],
        "coll_cos_before": [],
        "coll_cos_after": [],
        "full_cos_before": [],
        "full_cos_after": [],
        "block_polue_cos_before": [],
        "block_polue_cos_after": [],
        "gain_min": [],
        "gain_max": [],
        "delta_num": 0.0,
        "delta_den": 0.0,
    }
    by_regime = {r: {"count": 0, "eta_coll": [], "eta_full": [], "coll_after": [], "full_after": []} for r in REGIMES}

    for start in range(0, base.shape[0], int(args.project_batch)):
        end = min(start + int(args.project_batch), base.shape[0])
        n = end - start
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        p0 = (ad0.real**2 + ad0.imag**2).astype(np.float32)
        total0 = np.maximum(p0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        cur_full = p0 / total0[:, None, None, None, None, None]
        cur_coll_raw = p0.sum(axis=(1, 4))
        cur_coll = cur_coll_raw / total0[:, None, None, None]

        tgt_full = np.asarray(full_target[start:end], dtype=np.float32)
        tgt_coll = np.asarray(coll_target[start:end], dtype=np.float32)
        ratio_coll = ((tgt_coll + float(args.target_floor)) / (cur_coll + float(args.target_floor))) ** eta_coll[start:end, None, None, None]
        ratio_full = ((tgt_full + float(args.target_floor)) / (cur_full + float(args.target_floor))) ** eta_full[
            start:end, None, None, None, None, None
        ]
        ratio = ratio_coll[:, None, :, :, None, :] * ratio_full
        gain = np.sqrt(ratio).astype(np.float32)
        gain = np.clip(gain, gain_clip[0], gain_clip[1])

        flat_power = cur_coll_raw.reshape(n, -1)
        ktop = min(int(args.top_components), flat_power.shape[1])
        top_ids = np.argpartition(flat_power, -ktop, axis=1)[:, -ktop:]
        top_mask = np.zeros_like(flat_power, dtype=bool)
        top_mask[np.arange(n)[:, None], top_ids] = True
        top_mask = top_mask.reshape(n, MH, MV, S_SC)[:, None, :, :, None, :]
        gain = np.where(top_mask, np.clip(gain, top_gain_clip[0], top_gain_clip[1]), gain)

        ad1 = (ad0 * gain).astype(np.complex64)
        p1 = (ad1.real**2 + ad1.imag**2).astype(np.float32)
        total1 = np.maximum(p1.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        ad1 *= np.sqrt(total0 / total1)[:, None, None, None, None, None].astype(np.float32)
        h1 = inverse_ad(ad1).reshape(n, 256, 4, 192)
        h1 = h1 * np.exp(-1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        out[start:end] = h1.astype(np.complex64)
        out.flush()

        p1 = (ad1.real**2 + ad1.imag**2).astype(np.float32)
        total_after = np.maximum(p1.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        full1 = p1 / total_after[:, None, None, None, None, None]
        coll1 = p1.sum(axis=(1, 4)) / total_after[:, None, None, None]
        stats["power_ratio"].extend((total_after / total0).tolist())
        stats["coll_cos_before"].extend(row_cos(cur_coll, tgt_coll).tolist())
        stats["coll_cos_after"].extend(row_cos(coll1, tgt_coll).tolist())
        stats["full_cos_before"].extend(row_cos(cur_full, tgt_full).tolist())
        stats["full_cos_after"].extend(row_cos(full1, tgt_full).tolist())
        # Per-pol/per-UE block support shape: B,pol,UE,H,V,T cosine averaged.
        cur_block = cur_full.transpose(0, 1, 4, 2, 3, 5).reshape(n * MP * N_ANT, MH * MV * S_SC)
        tgt_block = tgt_full.transpose(0, 1, 4, 2, 3, 5).reshape(n * MP * N_ANT, MH * MV * S_SC)
        one_block = full1.transpose(0, 1, 4, 2, 3, 5).reshape(n * MP * N_ANT, MH * MV * S_SC)
        block_before = row_cos(cur_block, tgt_block).reshape(n, MP * N_ANT).mean(axis=1)
        block_after = row_cos(one_block, tgt_block).reshape(n, MP * N_ANT).mean(axis=1)
        stats["block_polue_cos_before"].extend(block_before.tolist())
        stats["block_polue_cos_after"].extend(block_after.tolist())
        stats["gain_min"].extend(np.min(gain.reshape(n, -1), axis=1).tolist())
        stats["gain_max"].extend(np.max(gain.reshape(n, -1), axis=1).tolist())
        diff = h1 - h0
        stats["delta_num"] += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
        stats["delta_den"] += float(np.sum(np.abs(h0) ** 2, dtype=np.float64))
        for j, reg in enumerate(regimes[start:end]):
            b = by_regime[str(reg)]
            b["count"] += 1
            b["eta_coll"].append(float(eta_coll[start + j]))
            b["eta_full"].append(float(eta_full[start + j]))
            b["coll_after"].append(float(stats["coll_cos_after"][-n + j]))
            b["full_after"].append(float(stats["full_cos_after"][-n + j]))
        print(f"projected {end}/500", flush=True)

    return {
        "eta_collapsed": summarize(stats["eta_collapsed"]),
        "eta_full": summarize(stats["eta_full"]),
        "knn_max": summarize(stats["knn_max"]),
        "knn_entropy": summarize(stats["knn_entropy"]),
        "confidence": summarize(stats["confidence"]),
        "power_ratio": summarize(stats["power_ratio"]),
        "coll_cos_before": summarize(stats["coll_cos_before"]),
        "coll_cos_after": summarize(stats["coll_cos_after"]),
        "full_cos_before": summarize(stats["full_cos_before"]),
        "full_cos_after": summarize(stats["full_cos_after"]),
        "block_polue_cos_before": summarize(stats["block_polue_cos_before"]),
        "block_polue_cos_after": summarize(stats["block_polue_cos_after"]),
        "gain_min": summarize(stats["gain_min"]),
        "gain_max": summarize(stats["gain_max"]),
        "delta_nmse_to_base": float(stats["delta_num"] / max(stats["delta_den"], TINY)),
        "by_regime": {r: {k: summarize(v) if isinstance(v, list) else v for k, v in vals.items()} for r, vals in by_regime.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v115/v115_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v119/v119_Round1_Test_Channel.npy")
    ap.add_argument("--train-base-ad", default=ASSETS + "/train_v36_loo_base_derot_ad.npy")
    ap.add_argument("--train-residual-ad", default=ASSETS + "/train_v36_loo_residual_derot_ad.npy")
    ap.add_argument("--current-target", default=ASSETS + "/cache_v115/v115_pointcloud_scene_support_target.npy")
    ap.add_argument("--anchor-v112", default=ASSETS + "/cache_v112/v112_kernel_moe_path_support_target.npy")
    ap.add_argument("--full-target-out", default=ASSETS + "/cache_v119/v119_full_metric_block_target.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v119/v119_hybrid_metric_collapsed_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v119/v119_hybrid_metric_weights.npy")
    ap.add_argument("--report", default=ASSETS + "/v119_hybrid_metric_aligned_support_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v119_hybrid_metric_aligned_support_candidate_pending.zip"))
    ap.add_argument("--candidate-zip", default=str(PATHS.round1_versions / "Round1_Submission_candidate_v119_hybrid_metric_aligned_support.zip"))
    ap.add_argument("--k-full", type=int, default=12)
    ap.add_argument("--full-power", type=float, default=3.00)
    ap.add_argument("--full-floor-dist", type=float, default=0.55)
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--target-floor", type=float, default=1e-12)
    ap.add_argument("--build-batch", type=int, default=8)
    ap.add_argument("--project-batch", type=int, default=8)
    ap.add_argument("--eta-collapsed", default="clear:0.0003,light:0.0030,medium:0.420,heavy:0.560")
    ap.add_argument("--eta-full", default="clear:0.0000,light:0.0010,medium:0.095,heavy:0.145")
    ap.add_argument("--coll-full-mix", type=float, default=0.34)
    ap.add_argument("--anchor-v112-mix", type=float, default=0.035)
    ap.add_argument("--min-gate", type=float, default=0.28)
    ap.add_argument("--max-gate", type=float, default=2.20)
    ap.add_argument("--conf-center", type=float, default=0.18)
    ap.add_argument("--conf-width", type=float, default=0.16)
    ap.add_argument("--entropy-low", type=float, default=0.15)
    ap.add_argument("--entropy-high", type=float, default=0.88)
    ap.add_argument("--gain-clip", default="0.42,3.20")
    ap.add_argument("--top-gain-clip", default="0.55,2.30")
    ap.add_argument("--top-components", type=int, default=2048)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    args = ap.parse_args()

    data = Path(args.data)
    pos_train = np.load(data / "Round1_Train_Pos.npy")
    pos_test = np.load(data / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes_test = regime_of(blocked_test)

    scene = np.load(args.scene_cache, allow_pickle=True)
    scene_train = scene["train_features"].astype(np.float32)
    scene_test = scene["test_features"].astype(np.float32)
    wall_patches = scene["wall_patches"].astype(np.float32)
    train_embed, test_embed = standardize(
        official_scene_embedding(
            pos_train,
            blocked_train,
            scene_train,
            scene["train_reflection_score"].astype(np.float32),
            scene["train_reflection_path_length"].astype(np.float32),
            scene["train_reflection_excess_length"].astype(np.float32),
            scene["train_reflection_point"].astype(np.float32),
            scene["train_reflection_patch_id"].astype(np.int64),
            wall_patches,
        ),
        official_scene_embedding(
            pos_test,
            blocked_test,
            scene_test,
            scene["test_reflection_score"].astype(np.float32),
            scene["test_reflection_path_length"].astype(np.float32),
            scene["test_reflection_excess_length"].astype(np.float32),
            scene["test_reflection_point"].astype(np.float32),
            scene["test_reflection_patch_id"].astype(np.int64),
            wall_patches,
        ),
    )

    print("build official-scene full-support kNN", flush=True)
    idx, dist = pairwise_knn(test_embed, train_embed, int(args.k_full), exclude_self=False)
    weights = weights_from_dist(dist, float(args.full_power), float(args.full_floor_dist))
    Path(args.weights_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.weights_out, weights.astype(np.float32))

    print("build full per-pol/per-UE target", flush=True)
    full_target = full_log_average(
        args.train_base_ad,
        args.train_residual_ad,
        idx,
        weights,
        args.full_target_out,
        float(args.floor),
        int(args.build_batch),
    )
    full_coll = np.asarray(full_target.sum(axis=(1, 4)), dtype=np.float32)
    full_coll /= np.maximum(full_coll.sum(axis=(1, 2, 3), keepdims=True), TINY)

    current = normalize_support(np.load(args.current_target, mmap_mode="r"), float(args.floor))
    anchor_v112 = normalize_support(np.load(args.anchor_v112, mmap_mode="r"), float(args.floor))
    coeff = np.zeros((len(current), 3), dtype=np.float32)
    full_mix = float(args.coll_full_mix)
    anchor_mix = float(args.anchor_v112_mix)
    coeff[:, 0] = 1.0 - full_mix - anchor_mix
    coeff[:, 1] = full_mix
    coeff[:, 2] = anchor_mix
    coll_target = log_blend([current, full_coll, anchor_v112], coeff, float(args.floor), int(args.build_batch))
    np.save(args.target_out, coll_target.astype(np.float32))

    proj = project_hybrid(full_target, coll_target, weights, regimes_test, pos_test, args)
    report = {
        "method": "v119_hybrid_metric_aligned_support",
        "base": args.base,
        "operation": "v115 backbone; collapsed HVT target plus full per-pol/per-UE AD support target from official point-cloud kNN; phase/scalar/total power preserved",
        "args": vars(args),
        "knn": {
            "weight_max": summarize(weights.max(axis=1)),
            "weight_entropy": summarize(entropy01(weights)),
        },
        "target_cos": {
            "full_coll_vs_current": summarize(row_cos(full_coll, current)),
            "coll_target_vs_current": summarize(row_cos(coll_target, current)),
            "coll_target_vs_full_coll": summarize(row_cos(coll_target, full_coll)),
            "coll_target_vs_v112": summarize(row_cos(coll_target, anchor_v112)),
        },
        "projection": proj,
        "outputs": {
            "submission_npy": args.output,
            "full_target": args.full_target_out,
            "collapsed_target": args.target_out,
            "weights": args.weights_out,
        },
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    def write_zip(path: str) -> None:
        files = [
            (args.output, "Round1_Test_Channel.npy"),
            (__file__, "generate_v119_hybrid_metric_aligned_support.py"),
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
