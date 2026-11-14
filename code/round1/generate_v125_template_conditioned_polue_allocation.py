#!/usr/bin/env python3
"""Generate v125: template-conditioned polarization/UE allocation.

The v124 bottleneck oracle shows that total power and a common phase are not
the main remaining errors.  The large tractable gap is the energy allocation
inside each collapsed (horizontal angle, vertical angle, delay) component:
two BS polarizations times four UE ports.

This version learns that eight-way allocation from train truth, conditioned on
the 96 propagation-support templates discovered by v121.  The v124 soft
template probabilities select and blend those allocations for each test point.
Only positive real gains are applied in angle-delay space.  Every collapsed
H/V/delay component is renormalized to its original v124 energy, so the method
preserves:

* v124 collapsed joint AD support;
* per-point total power;
* complex phase of every coefficient.

No Sionna mesh/object/material labels or external submission residuals are
used.  The learned statistics come from official train channels and the
template selector comes from official positions and point-cloud geometry.
"""

from __future__ import annotations

import argparse
import json
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
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def entropy01(probabilities: np.ndarray) -> np.ndarray:
    p = np.maximum(probabilities.astype(np.float64), 1e-30)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-30)
    return (-np.sum(p * np.log(p), axis=1) / np.log(max(p.shape[1], 2))).astype(np.float32)


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


def normalized_power(ad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = (ad.real**2 + ad.imag**2).astype(np.float32)
    total = np.maximum(power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
    power /= total
    collapsed = power.sum(axis=(1, 4))
    return power, collapsed


def allocation_from_power(power: np.ndarray, collapsed: np.ndarray, floor: float) -> np.ndarray:
    return power / np.maximum(collapsed[:, None, :, :, None, :], float(floor))


def build_template_allocations(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    labels = np.load(args.labels).astype(np.int64)
    clusters = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=clusters).astype(np.int64)
    base_ad = np.load(args.train_base_ad, mmap_mode="r")
    residual_ad = np.load(args.train_residual_ad, mmap_mode="r")
    if len(labels) != len(base_ad) or base_ad.shape != residual_ad.shape:
        raise ValueError("train labels/base/residual shapes are inconsistent")

    component_shape = (MP, MH, MV, N_ANT, S_SC)
    template_power = np.zeros((clusters,) + component_shape, dtype=np.float64)
    template_collapsed = np.zeros((clusters, MH, MV, S_SC), dtype=np.float64)
    global_power = np.zeros(component_shape, dtype=np.float64)
    global_collapsed = np.zeros((MH, MV, S_SC), dtype=np.float64)

    print("build template-conditioned pol/UE allocations", flush=True)
    for start in range(0, len(labels), int(args.train_batch)):
        end = min(start + int(args.train_batch), len(labels))
        ad = np.array(base_ad[start:end], dtype=np.complex64, copy=True)
        ad += np.asarray(residual_ad[start:end], dtype=np.complex64)
        power, collapsed = normalized_power(ad)
        global_power += power.sum(axis=0, dtype=np.float64)
        global_collapsed += collapsed.sum(axis=0, dtype=np.float64)
        batch_labels = labels[start:end]
        for cluster in np.unique(batch_labels):
            mask = batch_labels == cluster
            template_power[int(cluster)] += power[mask].sum(axis=0, dtype=np.float64)
            template_collapsed[int(cluster)] += collapsed[mask].sum(axis=0, dtype=np.float64)
        print(f"template stats {end}/{len(labels)}", flush=True)

    global_alloc = global_power / np.maximum(global_collapsed[None, :, :, None, :], float(args.alloc_floor))
    global_alloc /= np.maximum(global_alloc.sum(axis=(0, 3), keepdims=True), float(args.alloc_floor))

    prior = float(args.prior_points)
    mean_power = global_power / max(float(len(labels)), 1.0)
    mean_collapsed = global_collapsed / max(float(len(labels)), 1.0)
    template_power += prior * mean_power[None, ...]
    template_collapsed += prior * mean_collapsed[None, ...]
    allocations = template_power / np.maximum(
        template_collapsed[:, None, :, :, None, :], float(args.alloc_floor)
    )
    allocations /= np.maximum(allocations.sum(axis=(1, 4), keepdims=True), float(args.alloc_floor))

    # The final cache is float32 (about 75 MB for 96 templates) while all
    # aggregation above stays float64 to avoid losing low-power components.
    allocations = allocations.astype(np.float32)
    Path(args.allocation_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.allocation_out, allocations)

    global_flat = np.broadcast_to(global_alloc[None, ...], allocations.shape)
    alloc_cos_global = row_cos(allocations, global_flat)
    report = {
        "clusters": clusters,
        "cluster_count": summarize(counts),
        "prior_points": prior,
        "template_allocation_vs_global": summarize(alloc_cos_global),
        "allocation_cache": args.allocation_out,
    }
    return allocations, report


def blend_allocations(
    allocations: np.ndarray,
    probabilities: np.ndarray,
    top_templates: int,
    batch: int,
    floor: float,
) -> np.ndarray:
    """Blend template allocations in log space with sparse template weights."""

    n, clusters = probabilities.shape
    top = min(int(top_templates), clusters)
    idx = np.argpartition(probabilities, -top, axis=1)[:, -top:]
    weights = np.take_along_axis(probabilities, idx, axis=1).astype(np.float64)
    order = np.argsort(-weights, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    weights = np.take_along_axis(weights, order, axis=1)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)

    shape = (n, MP, MH, MV, N_ANT, S_SC)
    out = np.empty(shape, dtype=np.float32)
    for start in range(0, n, int(batch)):
        end = min(start + int(batch), n)
        acc = np.zeros((end - start, MP, MH, MV, N_ANT, S_SC), dtype=np.float64)
        for j in range(top):
            alloc = allocations[idx[start:end, j]].astype(np.float64)
            acc += weights[start:end, j, None, None, None, None, None] * np.log(np.maximum(alloc, floor))
        val = np.exp(acc)
        val /= np.maximum(val.sum(axis=(1, 4), keepdims=True), floor)
        out[start:end] = val.astype(np.float32)
        print(f"allocation target {end}/{n}", flush=True)
    return out


def project_allocation(
    target: np.ndarray,
    probabilities: np.ndarray,
    regimes: np.ndarray,
    pos_test: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    base = np.load(args.base, mmap_mode="r")
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)
    beta_s = beta_subcarriers(float(args.beta), float(args.gamma))
    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    eta_by_regime = parse_regime_values(args.eta, 0.0)

    top1 = probabilities.max(axis=1).astype(np.float32)
    ent = entropy01(probabilities).astype(np.float32)
    confidence = np.clip((top1 - float(args.conf_center)) / max(float(args.conf_width), 1e-6), 0.0, 1.0)
    confidence *= np.clip(
        (float(args.entropy_high) - ent) / max(float(args.entropy_high) - float(args.entropy_low), 1e-6),
        0.0,
        1.0,
    )
    gate = float(args.min_gate) + (float(args.max_gate) - float(args.min_gate)) * confidence
    eta = np.asarray([eta_by_regime[str(r)] for r in regimes], dtype=np.float32) * gate.astype(np.float32)
    gain_clip = tuple(float(x) for x in args.gain_clip.split(","))
    top_gain_clip = tuple(float(x) for x in args.top_gain_clip.split(","))

    stats: dict[str, object] = {
        "eta": [],
        "top1": top1.tolist(),
        "entropy": ent.tolist(),
        "confidence": confidence.tolist(),
        "allocation_cos_before": [],
        "allocation_cos_after": [],
        "block_cos_before": [],
        "block_cos_after": [],
        "collapsed_cos": [],
        "power_ratio": [],
        "gain_min": [],
        "gain_max": [],
        "delta_num": 0.0,
        "delta_den": 0.0,
    }
    by_regime = {
        r: {"count": 0, "eta": [], "allocation_before": [], "allocation_after": []}
        for r in REGIMES
    }

    for start in range(0, len(base), int(args.project_batch)):
        end = min(start + int(args.project_batch), len(base))
        n = end - start
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        p0 = (ad0.real**2 + ad0.imag**2).astype(np.float32)
        coll0 = p0.sum(axis=(1, 4))
        total0 = np.maximum(coll0.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        cur_alloc = allocation_from_power(p0, coll0, float(args.alloc_floor))
        tgt = np.asarray(target[start:end], dtype=np.float32)

        ratio = ((tgt + float(args.ratio_floor)) / (cur_alloc + float(args.ratio_floor))) ** eta[
            start:end, None, None, None, None, None
        ]
        gain = np.sqrt(ratio).astype(np.float32)
        gain = np.clip(gain, gain_clip[0], gain_clip[1])

        flat_coll = coll0.reshape(n, -1)
        ktop = min(int(args.top_components), flat_coll.shape[1])
        top_ids = np.argpartition(flat_coll, -ktop, axis=1)[:, -ktop:]
        top_mask = np.zeros_like(flat_coll, dtype=bool)
        top_mask[np.arange(n)[:, None], top_ids] = True
        top_mask = top_mask.reshape(n, MH, MV, S_SC)[:, None, :, :, None, :]
        # Only high-energy components receive the learned correction.  The
        # low-energy tail is too noisy to estimate an eight-way allocation.
        gain = np.where(top_mask, np.clip(gain, top_gain_clip[0], top_gain_clip[1]), 1.0)

        ad1 = (ad0 * gain).astype(np.complex64)
        p1 = (ad1.real**2 + ad1.imag**2).astype(np.float32)
        coll1 = np.maximum(p1.sum(axis=(1, 4)), TINY)
        # Exact per-HVT renormalization: this is what guarantees that v124's
        # collapsed support and total power are unchanged up to roundoff.
        ad1 *= np.sqrt(coll0 / coll1)[:, None, :, :, None, :].astype(np.float32)

        p1 = (ad1.real**2 + ad1.imag**2).astype(np.float32)
        coll1 = p1.sum(axis=(1, 4))
        total1 = np.maximum(coll1.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        h1 = inverse_ad(ad1).reshape(n, 256, 4, 192)
        h1 *= np.exp(-1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite output block {start}:{end}")
        out[start:end] = h1.astype(np.complex64)
        out.flush()

        alloc1 = allocation_from_power(p1, coll1, float(args.alloc_floor))
        stats["eta"].extend(eta[start:end].tolist())
        stats["allocation_cos_before"].extend(row_cos(cur_alloc, tgt).tolist())
        stats["allocation_cos_after"].extend(row_cos(alloc1, tgt).tolist())
        cur_block = p0.transpose(0, 1, 4, 2, 3, 5).reshape(n * MP * N_ANT, -1)
        one_block = p1.transpose(0, 1, 4, 2, 3, 5).reshape(n * MP * N_ANT, -1)
        tgt_block = (
            tgt * coll0[:, None, :, :, None, :]
        ).transpose(0, 1, 4, 2, 3, 5).reshape(n * MP * N_ANT, -1)
        stats["block_cos_before"].extend(row_cos(cur_block, tgt_block).reshape(n, MP * N_ANT).mean(axis=1).tolist())
        stats["block_cos_after"].extend(row_cos(one_block, tgt_block).reshape(n, MP * N_ANT).mean(axis=1).tolist())
        stats["collapsed_cos"].extend(row_cos(coll0, coll1).tolist())
        stats["power_ratio"].extend((total1 / total0).tolist())
        stats["gain_min"].extend(gain.reshape(n, -1).min(axis=1).tolist())
        stats["gain_max"].extend(gain.reshape(n, -1).max(axis=1).tolist())
        diff = h1 - h0
        stats["delta_num"] += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
        stats["delta_den"] += float(np.sum(np.abs(h0) ** 2, dtype=np.float64))
        before = stats["allocation_cos_before"][-n:]
        after = stats["allocation_cos_after"][-n:]
        for j, reg in enumerate(regimes[start:end]):
            item = by_regime[str(reg)]
            item["count"] += 1
            item["eta"].append(float(eta[start + j]))
            item["allocation_before"].append(float(before[j]))
            item["allocation_after"].append(float(after[j]))
        print(f"project allocation {end}/{len(base)}", flush=True)

    return {
        "eta": summarize(stats["eta"]),
        "template_probability_top1": summarize(stats["top1"]),
        "template_probability_entropy": summarize(stats["entropy"]),
        "confidence": summarize(stats["confidence"]),
        "allocation_cos_before": summarize(stats["allocation_cos_before"]),
        "allocation_cos_after": summarize(stats["allocation_cos_after"]),
        "block_cos_before": summarize(stats["block_cos_before"]),
        "block_cos_after": summarize(stats["block_cos_after"]),
        "collapsed_cos_preserved": summarize(stats["collapsed_cos"]),
        "power_ratio": summarize(stats["power_ratio"]),
        "gain_min": summarize(stats["gain_min"]),
        "gain_max": summarize(stats["gain_max"]),
        "delta_nmse_to_v124": float(stats["delta_num"] / max(float(stats["delta_den"]), TINY)),
        "by_regime": {
            r: {k: summarize(v) if isinstance(v, list) else v for k, v in item.items()}
            for r, item in by_regime.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--base", default=ASSETS + "/cache_v124/v124_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v125/v125_Round1_Test_Channel.npy")
    ap.add_argument("--train-base-ad", default=ASSETS + "/train_v98like_loo_base_ad.npy")
    ap.add_argument("--train-residual-ad", default=ASSETS + "/train_v98like_loo_residual_ad.npy")
    ap.add_argument("--labels", default=ASSETS + "/cache_v121/v121_support_template_manifold_train_labels.npy")
    ap.add_argument("--probabilities", default=ASSETS + "/cache_v124/v124_visibility_misleading_overdrive_probs.npy")
    ap.add_argument("--allocation-out", default=ASSETS + "/cache_v125/v125_template_polue_allocations.npy")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v125/v125_template_polue_target.npy")
    ap.add_argument("--report", default=ASSETS + "/v125_template_conditioned_polue_allocation_generate.json")
    ap.add_argument(
        "--zip-out",
        default=str(PATHS.round1_versions / "v125_template_conditioned_polue_allocation_candidate_pending.zip"),
    )
    ap.add_argument("--prior-points", type=float, default=12.0)
    ap.add_argument("--rebuild-allocations", action="store_true")
    ap.add_argument("--top-templates", type=int, default=12)
    ap.add_argument("--eta", default="clear:0.004,light:0.018,medium:0.260,heavy:0.360")
    ap.add_argument("--min-gate", type=float, default=0.45)
    ap.add_argument("--max-gate", type=float, default=1.35)
    ap.add_argument("--conf-center", type=float, default=0.18)
    ap.add_argument("--conf-width", type=float, default=0.34)
    ap.add_argument("--entropy-low", type=float, default=0.18)
    ap.add_argument("--entropy-high", type=float, default=0.96)
    ap.add_argument("--gain-clip", default="0.62,1.62")
    ap.add_argument("--top-gain-clip", default="0.58,1.75")
    ap.add_argument("--top-components", type=int, default=1536)
    ap.add_argument("--alloc-floor", type=float, default=1e-10)
    ap.add_argument("--ratio-floor", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--blend-batch", type=int, default=4)
    ap.add_argument("--project-batch", type=int, default=8)
    args = ap.parse_args()

    data = Path(args.data)
    pos_test = np.load(data / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    regimes = regime_of(geo["test_blocked"].astype(np.float64))
    probabilities = np.load(args.probabilities).astype(np.float32)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), TINY)

    allocation_path = Path(args.allocation_out)
    if allocation_path.exists() and not args.rebuild_allocations:
        allocations = np.load(allocation_path, mmap_mode="r")
        labels = np.load(args.labels).astype(np.int64)
        template_report = {
            "clusters": int(allocations.shape[0]),
            "cluster_count": summarize(np.bincount(labels, minlength=allocations.shape[0])),
            "prior_points": float(args.prior_points),
            "allocation_cache": args.allocation_out,
            "reused": True,
        }
    else:
        allocations, template_report = build_template_allocations(args)

    target = blend_allocations(
        allocations,
        probabilities,
        int(args.top_templates),
        int(args.blend_batch),
        float(args.alloc_floor),
    )
    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target.astype(np.float32))
    projection = project_allocation(target, probabilities, regimes, pos_test, args)

    payload = {
        "method": "v125_template_conditioned_polue_allocation",
        "base": args.base,
        "operation": (
            "learn template-conditioned 2-polarization x 4-UE-port energy allocation inside every H/V/delay "
            "component; select by v124 soft propagation templates; preserve v124 collapsed AD support, total power, "
            "and complex phase exactly"
        ),
        "args": vars(args),
        "template_allocation": template_report,
        "projection": projection,
        "outputs": {
            "submission_npy": args.output,
            "allocation_templates": args.allocation_out,
            "allocation_target": args.target_out,
        },
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    files = [
        (args.output, "Round1_Test_Channel.npy"),
        (__file__, "generate_v125_template_conditioned_polue_allocation.py"),
        (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
        (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
        (PATHS.code_root / "project_paths.py", "project_paths.py"),
    ]
    Path(args.zip_out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_STORED) as zf:
        for src, arc in files:
            zf.write(src, arc)
    print(f"wrote {args.zip_out}", flush=True)


if __name__ == "__main__":
    main()
