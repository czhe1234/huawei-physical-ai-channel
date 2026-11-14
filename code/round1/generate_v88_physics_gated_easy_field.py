#!/usr/bin/env python3
"""Generate v88: physics/confidence gated easy-field projection.

v87 proved that projecting easy non-coherent targets on top of the hard
baseline can improve the online score.  v88 keeps the same idea but makes the
projection strength point-wise:

  - neighbor PAS/PDP agreement gates the easy-field trust;
  - Sionna path-delay confidence gates light/boundary/reflection cases;
  - UE/polarization ratios get slightly stronger correction because they are
    highly stable in LOO diagnostics;
  - total power remains off by default because LOO total-power tails are large.

The output is still produced only by positive real amplitude gains in PHV
angle-delay domain.  Phase and component positions are not directly changed.
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


def path_delay_confidence(path_delay: np.ndarray, path_valid: np.ndarray, sigma: float) -> np.ndarray:
    delay_grid = np.arange(S_SC, dtype=np.float32)[None, :]
    conf = np.zeros((path_delay.shape[0], S_SC), dtype=np.float32)
    for i in range(path_delay.shape[0]):
        valid = path_valid[i] & np.isfinite(path_delay[i])
        if not np.any(valid):
            continue
        err = np.min(np.abs(delay_grid - path_delay[i, valid].astype(np.float32)[:, None]), axis=0)
        conf[i] = np.exp(-0.5 * (err / max(float(sigma), 1e-6)) ** 2).astype(np.float32)
    return conf


def load_path_features(path_file: str) -> dict[str, np.ndarray]:
    paths = np.load(path_file)
    valid = paths["test_valid"].astype(bool)
    delay = paths["test_delay_bin"].astype(np.float32)
    ptype = paths["test_path_type"].astype(np.int8)
    score = paths["test_score"].astype(np.float32)
    los_clear = paths["test_los_clear"].astype(bool)
    los_hit_count = paths["test_los_hit_count"].astype(np.float32)
    ref_valid = valid & (ptype == 1)
    ref_score = np.where(ref_valid, score, -np.inf)
    best_ref_score = np.max(ref_score, axis=1)
    best_ref_score = np.where(np.isfinite(best_ref_score), best_ref_score, 0.0).astype(np.float32)
    ref_gate = np.clip((best_ref_score - 2.9) / 0.5, 0.0, 1.0).astype(np.float32)
    hit_gate = np.clip((los_hit_count - 2.0) / 10.0, 0.0, 1.0).astype(np.float32)
    return {
        "valid": valid,
        "delay": delay,
        "type": ptype,
        "los_clear": los_clear,
        "hit_gate": hit_gate,
        "ref_gate": ref_gate,
        "all_delay_conf": path_delay_confidence(delay, valid, sigma=0.75),
        "los_delay_conf": path_delay_confidence(delay, valid & (ptype == 0), sigma=0.75),
        "ref_delay_conf": path_delay_confidence(delay, valid & (ptype == 1), sigma=0.75),
    }


def soft_gate(x: np.ndarray, lo: float, hi: float, power: float = 1.0) -> np.ndarray:
    g = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if power != 1.0:
        g = g ** float(power)
    return g.astype(np.float32)


def pointwise_etas(
    regimes: np.ndarray,
    targets: dict[str, np.ndarray],
    path: dict[str, np.ndarray],
    top_path_conf: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(regimes)
    beam_ag = targets["agree_beam"].astype(np.float32)
    delay_ag = targets["agree_delay"].astype(np.float32)
    beam_gate = soft_gate(beam_ag, 0.42, 0.86)
    delay_gate = soft_gate(delay_ag, 0.58, 0.95)
    joint_gate = np.sqrt(np.maximum(beam_gate * delay_gate, 0.0)).astype(np.float32)
    path_gate = soft_gate(top_path_conf, 0.25, 0.75)

    out = {k: np.zeros(n, dtype=np.float32) for k in ["beam", "delay", "pol", "ue", "top"]}
    for i, reg in enumerate(regimes):
        r = str(reg)
        g = float(joint_gate[i])
        pg = float(path_gate[i])
        ref = float(path["ref_gate"][i])
        hit = float(path["hit_gate"][i])
        los_like = bool(path["los_clear"][i])

        if r == "clear":
            # Clear PAS/PDP is accurate in LOO, but v83 is already good.
            out["beam"][i] = 0.035 * (0.45 + 0.75 * beam_gate[i])
            out["delay"][i] = 0.025 * (0.45 + 0.75 * delay_gate[i])
            out["pol"][i] = 0.045
            out["ue"][i] = 0.045
            out["top"][i] = 0.0
        elif r == "light":
            # Light is boundary-like: only trust PAS/PDP if both local spectra
            # and path delay agree.  UE/pol is stable enough to use generally.
            stable = g * (0.45 + 0.55 * pg)
            if los_like:
                stable = max(stable, 0.35 * float(delay_gate[i]))
            out["beam"][i] = 0.10 * stable
            out["delay"][i] = 0.075 * stable
            out["pol"][i] = 0.060 * (0.7 + 0.3 * g)
            out["ue"][i] = 0.060 * (0.7 + 0.3 * g)
            out["top"][i] = 0.050 * stable
        elif r == "medium":
            # Medium usually has useful PAS/PDP signal; gate weak points down
            # and high-confidence reflection/LOS-consistent points up.
            phys = 0.75 + 0.25 * pg + 0.10 * ref
            out["beam"][i] = 0.18 * (0.45 + 0.95 * g) * phys
            out["delay"][i] = 0.12 * (0.50 + 0.85 * delay_gate[i]) * phys
            out["pol"][i] = 0.125 * (0.80 + 0.25 * g)
            out["ue"][i] = 0.125 * (0.80 + 0.25 * g)
            out["top"][i] = 0.14 * (0.35 + 0.90 * pg) * (0.65 + 0.35 * g)
        else:
            # Heavy produced most of the v87 gain.  Emphasize stable reflection
            # cases, but cut projection on ambiguous low-agreement points.
            phys = 0.70 + 0.30 * pg + 0.15 * ref + 0.05 * hit
            out["beam"][i] = 0.18 * (0.40 + 1.00 * g) * phys
            out["delay"][i] = 0.085 * (0.45 + 0.85 * delay_gate[i]) * phys
            out["pol"][i] = 0.13 * (0.78 + 0.28 * g)
            out["ue"][i] = 0.13 * (0.78 + 0.28 * g)
            out["top"][i] = 0.13 * (0.35 + 0.95 * pg) * (0.60 + 0.40 * g)

    # Keep the experiment bounded.
    out["beam"] = np.clip(out["beam"], 0.0, 0.24)
    out["delay"] = np.clip(out["delay"], 0.0, 0.16)
    out["pol"] = np.clip(out["pol"], 0.0, 0.16)
    out["ue"] = np.clip(out["ue"], 0.0, 0.16)
    out["top"] = np.clip(out["top"], 0.0, 0.16)
    out["beam_gate"] = beam_gate
    out["delay_gate"] = delay_gate
    out["joint_gate"] = joint_gate
    out["path_gate"] = path_gate
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=ASSETS + "/cache_v83/v83_Round1_Test_Channel.npy")
    parser.add_argument("--output", default=str(PATHS.cache_dir(88) / "v88_Round1_Test_Channel.npy"))
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    parser.add_argument("--easy-cache", default=ASSETS + "/train_easy_marginals_v87_g010.npz")
    parser.add_argument("--sionna-paths", default=ASSETS + "/sionna_path_candidates.npz")
    parser.add_argument("--report", default=ASSETS + "/v88_physics_gated_easy_field_generate.json")
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--power", type=float, default=3.0)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--gain-clip-rest", default="0.75,1.25")
    parser.add_argument("--gain-clip-top16", default="0.92,1.12")
    parser.add_argument("--gain-clip-top4", default="0.97,1.05")
    parser.add_argument("--top-delta-clip", type=float, default=0.018)
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
    print("predicting v88 easy targets ...", flush=True)
    targets = predict_easy_targets(bank, pos_train, blocked_train, pos_test, blocked_test, args.k, args.power)
    path = load_path_features(args.sionna_paths)

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
        "top_path_conf": [],
        "eta_beam": [],
        "eta_delay": [],
        "eta_pol": [],
        "eta_ue": [],
        "eta_top": [],
    }
    by_regime = {
        r: {"power_ratio": [], "pdp_cos_to_base": [], "pas_cos_to_base": [], "top4_frac_delta": [], "eta_beam": []}
        for r in ["clear", "light", "medium", "heavy"]
    }

    top_path_conf_all = np.zeros(500, dtype=np.float32)
    etas: dict[str, np.ndarray] | None = None

    for start in range(0, 500, args.batch):
        end = min(start + args.batch, 500)
        n = end - start
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        p0 = (np.abs(ad0) ** 2).astype(np.float32)
        p = p0.copy()

        p0_coll = p0.sum(axis=(1, 4))
        mask4, mask16 = top_masks_from_collapsed(p0_coll, 4, 16)
        flat0 = p0_coll.reshape(n, -1)
        flat_mask4 = mask4.reshape(n, -1)
        path_conf_hvt = np.broadcast_to(path["all_delay_conf"][start:end, None, None, :], p0_coll.shape)
        top_den = np.maximum(np.sum(flat0 * flat_mask4, axis=1), TINY)
        top_path_conf = np.sum(flat0 * flat_mask4 * path_conf_hvt.reshape(n, -1), axis=1) / top_den
        top_path_conf_all[start:end] = top_path_conf.astype(np.float32)

        if etas is None:
            # Need top_path_conf for every point before pointwise_etas.  Fill
            # all points with a cheap pass here, then restart this block.
            tmp = np.zeros(500, dtype=np.float32)
            tmp[start:end] = top_path_conf.astype(np.float32)
            for s2 in range(end, 500, args.batch):
                e2 = min(s2 + args.batch, 500)
                hb = np.asarray(base[s2:e2], dtype=np.complex64)
                der = hb * np.exp(1j * d_test[s2:e2, None, None, None] * beta_s).astype(np.complex64)
                ad = ad_transform(der)
                pc = (np.abs(ad) ** 2).sum(axis=(1, 4)).astype(np.float32)
                m4, _m16 = top_masks_from_collapsed(pc, 4, 16)
                fl = pc.reshape(e2 - s2, -1)
                fm = m4.reshape(e2 - s2, -1)
                ph = np.broadcast_to(path["all_delay_conf"][s2:e2, None, None, :], pc.shape)
                den = np.maximum(np.sum(fl * fm, axis=1), TINY)
                tmp[s2:e2] = (np.sum(fl * fm * ph.reshape(e2 - s2, -1), axis=1) / den).astype(np.float32)
            top_path_conf_all[:] = tmp
            etas = pointwise_etas(regimes, targets, path, top_path_conf_all)

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

        p_coll = p.sum(axis=(1, 4))
        flat_coll = p_coll.reshape(n, -1)
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
        flat1 = p1_coll.reshape(n, -1)
        total1 = np.maximum(flat1.sum(axis=1), TINY)
        total0 = np.maximum(flat0.sum(axis=1), TINY)
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
        stats["top_path_conf"].extend(top_path_conf.tolist())
        for k in ["beam", "delay", "pol", "ue", "top"]:
            stats[f"eta_{k}"].extend(etas[k][start:end].tolist())

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
        print(f"generated {end}/500", flush=True)

    payload = {
        "method": "v88_physics_gated_easy_field",
        "base": args.base,
        "operation": "v87 easy-field projection with agreement and Sionna path-delay confidence gates; phase frozen",
        "neighbor": {"k": int(args.k), "power": float(args.power)},
        "iters": int(args.iters),
        "clips": {"rest": clip_rest, "top16": clip_top16, "top4": clip_top4, "top_delta": float(args.top_delta_clip)},
        "regime_counts": {r: int(np.sum(regimes == r)) for r in ["clear", "light", "medium", "heavy"]},
        "target_agreement": {
            "beam": summarize(targets["agree_beam"]),
            "delay": summarize(targets["agree_delay"]),
            "eff_nn": summarize(targets["eff_nn"]),
        },
        "gate_summary": {k: summarize(v) for k, v in etas.items()} if etas is not None else {},
        "overall": {k: summarize(v) for k, v in stats.items()},
        "by_regime": {r: {k: summarize(v) for k, v in vals.items()} for r, vals in by_regime.items()},
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
