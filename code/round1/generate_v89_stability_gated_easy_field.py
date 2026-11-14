#!/usr/bin/env python3
"""Generate v89: stability-gated easy-field projection.

v87 showed that easy-field projection helps.  v88 added path-physics gates but
slightly underperformed v87, likely because the gate definition was not aligned
with the online error.  v89 uses a cleaner definition of "easy":

    stable point = local neighbors have consistent PAS/PDP shapes.

This deliberately does not treat clear/light/medium/heavy as simple/hard by
default.  Regime only sets a mild base prior; the dominant control is local
spectral stability.  Phase and AD component locations remain frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of
from generate_v87_easy_field_projection import (
    MH,
    MV,
    MP,
    N_ANT,
    S_SC,
    BS_POS,
    ad_transform,
    apply_ratio,
    beta_subcarriers,
    build_easy_cache,
    inverse_ad,
    marginal_stats_from_power,
    predict_easy_targets,
    row_cos,
    summarize,
    top_masks_from_collapsed,
)


def soft_gate(x: np.ndarray, lo: float, hi: float, power: float = 1.0) -> np.ndarray:
    g = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if power != 1.0:
        g = g ** float(power)
    return g.astype(np.float32)


def build_stability_etas(regimes: np.ndarray, targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    beam_ag = targets["agree_beam"].astype(np.float32)
    delay_ag = targets["agree_delay"].astype(np.float32)
    beam_gate = soft_gate(beam_ag, 0.42, 0.88)
    delay_gate = soft_gate(delay_ag, 0.58, 0.96)
    stable = np.sqrt(np.maximum(beam_gate * delay_gate, 0.0)).astype(np.float32)
    high = soft_gate(stable, 0.68, 0.92)

    n = len(regimes)
    eta = {k: np.zeros(n, dtype=np.float32) for k in ["beam", "delay", "pol", "ue", "top"]}
    for i, reg in enumerate(regimes):
        r = str(reg)
        s = float(stable[i])
        bg = float(beam_gate[i])
        dg = float(delay_gate[i])
        h = float(high[i])

        if r == "clear":
            # Only truly stable clear points get a small correction.  This is
            # intentionally lower than v88's always-on clear correction.
            eta["beam"][i] = 0.045 * h
            eta["delay"][i] = 0.035 * h
            eta["pol"][i] = 0.050 * h
            eta["ue"][i] = 0.050 * h
            eta["top"][i] = 0.0
        elif r == "light":
            # Light is often boundary-like.  Enable only high-stability points.
            eta["beam"][i] = 0.070 * h
            eta["delay"][i] = 0.055 * h
            eta["pol"][i] = 0.055 * (0.25 + 0.75 * h)
            eta["ue"][i] = 0.055 * (0.25 + 0.75 * h)
            eta["top"][i] = 0.020 * h
        elif r == "medium":
            # Around v87 strength on average, stronger for stable points and
            # weaker for ambiguous points.
            eta["beam"][i] = 0.16 * (0.62 + 0.70 * s)
            eta["delay"][i] = 0.10 * (0.70 + 0.55 * dg)
            eta["pol"][i] = 0.115 * (0.75 + 0.35 * s)
            eta["ue"][i] = 0.115 * (0.75 + 0.35 * s)
            eta["top"][i] = 0.060 * (0.25 + 0.75 * s)
        else:
            # Heavy benefited from v87.  Keep most of that strength, but reduce
            # top/PAS movement when local spectra are not stable.
            eta["beam"][i] = 0.14 * (0.70 + 0.75 * s)
            eta["delay"][i] = 0.06 * (0.75 + 0.65 * dg)
            eta["pol"][i] = 0.120 * (0.75 + 0.35 * s)
            eta["ue"][i] = 0.120 * (0.75 + 0.35 * s)
            eta["top"][i] = 0.050 * (0.20 + 0.85 * s)

    eta["beam"] = np.clip(eta["beam"], 0.0, 0.22)
    eta["delay"] = np.clip(eta["delay"], 0.0, 0.14)
    eta["pol"] = np.clip(eta["pol"], 0.0, 0.16)
    eta["ue"] = np.clip(eta["ue"], 0.0, 0.16)
    eta["top"] = np.clip(eta["top"], 0.0, 0.10)
    eta["beam_gate"] = beam_gate
    eta["delay_gate"] = delay_gate
    eta["stable"] = stable
    eta["high_stable"] = high
    return eta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=ASSETS + "/cache_v83/v83_Round1_Test_Channel.npy")
    parser.add_argument("--output", default=str(PATHS.cache_dir(89) / "v89_Round1_Test_Channel.npy"))
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    parser.add_argument("--easy-cache", default=ASSETS + "/train_easy_marginals_v87_g010.npz")
    parser.add_argument("--report", default=ASSETS + "/v89_stability_gated_easy_field_generate.json")
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--power", type=float, default=3.0)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--gain-clip-rest", default="0.75,1.25")
    parser.add_argument("--gain-clip-top16", default="0.93,1.10")
    parser.add_argument("--gain-clip-top4", default="0.98,1.04")
    parser.add_argument("--top-delta-clip", type=float, default=0.012)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()

    clip_rest = tuple(float(x) for x in args.gain_clip_rest.split(","))
    clip_top16 = tuple(float(x) for x in args.gain_clip_top16.split(","))
    clip_top4 = tuple(float(x) for x in args.gain_clip_top4.split(","))

    data_dir = Path(args.data)
    pos_train = np.load(data_dir / "Round1_Train_Pos.npy")
    pos_test = np.load(data_dir / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked_test)
    beta_s = beta_subcarriers(args.beta, args.gamma)

    bank = build_easy_cache(Path(args.easy_cache), data_dir, beta_s, args.batch)
    print("predicting v89 easy targets ...", flush=True)
    targets = predict_easy_targets(bank, pos_train, blocked_train, pos_test, blocked_test, args.k, args.power)
    etas = build_stability_etas(regimes, targets)

    base = np.load(args.base, mmap_mode="r")
    if base.shape != (500, 256, 4, 192) or base.dtype != np.complex64:
        raise ValueError(f"unexpected base shape/dtype {base.shape} {base.dtype}")

    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)

    stats = {
        "power_ratio": [],
        "pdp_cos_to_base": [],
        "pas_cos_to_base": [],
        "pol_cos_to_base": [],
        "ue_cos_to_base": [],
        "gain_min": [],
        "gain_max": [],
        "top4_frac_delta": [],
        "top16_frac_delta": [],
        "eta_beam": [],
        "eta_delay": [],
        "eta_pol": [],
        "eta_ue": [],
        "eta_top": [],
        "stable": [],
    }
    by_regime = {
        r: {"power_ratio": [], "pdp_cos_to_base": [], "pas_cos_to_base": [], "top4_frac_delta": [], "eta_beam": [], "stable": []}
        for r in ["clear", "light", "medium", "heavy"]
    }

    for start in range(0, 500, args.batch):
        end = min(start + args.batch, 500)
        n = end - start
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        p0 = (np.abs(ad0) ** 2).astype(np.float32)
        p = p0.copy()

        target = {
            "beam": targets["beam"][start:end].reshape(n, MH, MV).astype(np.float32),
            "delay": targets["delay"][start:end].astype(np.float32),
            "pol": targets["pol"][start:end].astype(np.float32),
            "ue": targets["ue"][start:end].astype(np.float32),
        }

        for _ in range(args.iters):
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["beam"], cur["beam"], (etas["beam"][start:end] / args.iters)[:, None, None], (2, 3), 1e-8 / (MH * MV))
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["delay"], cur["delay"], (etas["delay"][start:end] / args.iters)[:, None], (5,), 1e-8 / S_SC)
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["pol"], cur["pol"], (etas["pol"][start:end] / args.iters)[:, None], (1,), 1e-8 / MP)
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["ue"], cur["ue"], (etas["ue"][start:end] / args.iters)[:, None], (4,), 1e-8 / N_ANT)

        p0_coll = p0.sum(axis=(1, 4))
        p_coll = p.sum(axis=(1, 4))
        mask4, mask16 = top_masks_from_collapsed(p0_coll, 4, 16)
        flat_coll = p_coll.reshape(n, -1)
        flat_mask4 = mask4.reshape(n, -1)
        cur_total = np.maximum(flat_coll.sum(axis=1), TINY)
        cur4 = np.sum(flat_coll * flat_mask4, axis=1) / cur_total
        target4 = targets["top4"][start:end].astype(np.float32)
        desired4 = cur4 * np.maximum(target4 / np.maximum(cur4, TINY), TINY) ** etas["top"][start:end]
        desired4 = np.clip(desired4, cur4 - args.top_delta_clip, cur4 + args.top_delta_clip)
        desired4 = np.clip(desired4, 1e-4, 0.98)
        top_scale = desired4 / np.maximum(cur4, TINY)
        rest_scale = (1.0 - desired4) / np.maximum(1.0 - cur4, TINY)
        scale_hvt = np.where(mask4, top_scale[:, None, None, None], rest_scale[:, None, None, None])
        p *= scale_hvt[:, None, :, :, None, :].astype(np.float32)

        p0_total = np.maximum(p0.sum(axis=(1, 2, 3, 4, 5)), TINY)
        p_total = np.maximum(p.sum(axis=(1, 2, 3, 4, 5)), TINY)
        p *= (p0_total / p_total)[:, None, None, None, None, None].astype(np.float32)

        gain = np.sqrt(p / np.maximum(p0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, clip_rest[0], clip_rest[1])
        mask16_full = mask16[:, None, :, :, None, :]
        mask4_full = mask4[:, None, :, :, None, :]
        gain = np.where(mask16_full, np.clip(gain, clip_top16[0], clip_top16[1]), gain)
        gain = np.where(mask4_full, np.clip(gain, clip_top4[0], clip_top4[1]), gain)

        ad1 = (ad0 * gain).astype(np.complex64)
        h1 = inverse_ad(ad1).reshape(n, 256, 4, 192)
        h1 = h1 * np.exp(-1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        out[start:end] = h1.astype(np.complex64)
        out.flush()

        p1 = (np.abs(ad1) ** 2).astype(np.float32)
        m0 = marginal_stats_from_power(p0)
        m1 = marginal_stats_from_power(p1)
        p1_coll = p1.sum(axis=(1, 4))
        flat0 = p0_coll.reshape(n, -1)
        flat1 = p1_coll.reshape(n, -1)
        total0 = np.maximum(flat0.sum(axis=1), TINY)
        total1 = np.maximum(flat1.sum(axis=1), TINY)
        top4_0 = np.sum(flat0 * flat_mask4, axis=1) / total0
        top4_1 = np.sum(flat1 * flat_mask4, axis=1) / total1
        mask16_flat = mask16.reshape(n, -1)
        top16_0 = np.sum(flat0 * mask16_flat, axis=1) / total0
        top16_1 = np.sum(flat1 * mask16_flat, axis=1) / total1

        stats["power_ratio"].extend((m1["total"] / m0["total"]).tolist())
        stats["pdp_cos_to_base"].extend(row_cos(m1["delay"], m0["delay"]).tolist())
        stats["pas_cos_to_base"].extend(row_cos(m1["beam"], m0["beam"]).tolist())
        stats["pol_cos_to_base"].extend(row_cos(m1["pol"], m0["pol"]).tolist())
        stats["ue_cos_to_base"].extend(row_cos(m1["ue"], m0["ue"]).tolist())
        stats["gain_min"].extend(np.min(gain.reshape(n, -1), axis=1).tolist())
        stats["gain_max"].extend(np.max(gain.reshape(n, -1), axis=1).tolist())
        stats["top4_frac_delta"].extend((top4_1 - top4_0).tolist())
        stats["top16_frac_delta"].extend((top16_1 - top16_0).tolist())
        for k in ["beam", "delay", "pol", "ue", "top", "stable"]:
            stats[f"eta_{k}" if k != "stable" else "stable"].extend(etas[k][start:end].tolist())

        pdpc = row_cos(m1["delay"], m0["delay"])
        pasc = row_cos(m1["beam"], m0["beam"])
        pr = m1["total"] / m0["total"]
        d4 = top4_1 - top4_0
        for j, reg in enumerate(regimes[start:end]):
            b = by_regime[str(reg)]
            b["power_ratio"].append(float(pr[j]))
            b["pdp_cos_to_base"].append(float(pdpc[j]))
            b["pas_cos_to_base"].append(float(pasc[j]))
            b["top4_frac_delta"].append(float(d4[j]))
            b["eta_beam"].append(float(etas["beam"][start + j]))
            b["stable"].append(float(etas["stable"][start + j]))
        print(f"generated {end}/500", flush=True)

    payload = {
        "method": "v89_stability_gated_easy_field",
        "base": args.base,
        "operation": "v87 easy-field projection with stability-defined easy regions; phase frozen",
        "neighbor": {"k": int(args.k), "power": float(args.power)},
        "iters": int(args.iters),
        "clips": {"rest": clip_rest, "top16": clip_top16, "top4": clip_top4, "top_delta": float(args.top_delta_clip)},
        "regime_counts": {r: int(np.sum(regimes == r)) for r in ["clear", "light", "medium", "heavy"]},
        "target_agreement": {
            "beam": summarize(targets["agree_beam"]),
            "delay": summarize(targets["agree_delay"]),
            "eff_nn": summarize(targets["eff_nn"]),
        },
        "gate_summary": {k: summarize(v) for k, v in etas.items()},
        "overall": {k: summarize(v) for k, v in stats.items()},
        "by_regime": {r: {k: summarize(v) for k, v in vals.items()} for r, vals in by_regime.items()},
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
