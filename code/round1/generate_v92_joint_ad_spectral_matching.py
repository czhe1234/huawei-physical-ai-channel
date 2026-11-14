#!/usr/bin/env python3
"""Generate v92: joint angle-delay spectral matching on top of v91.

v87 matched only marginal spectra (PAS/PDP/pol/UE/top fraction).  Marginal
matching cannot decide which angle bin and delay bin belong to the same path.
v92 estimates a full collapsed joint AD power distribution

    P(angle_h, angle_v, delay)

from nearby training points and multiplicatively projects the current v91 AD
power toward that joint support.  Phase is kept from v91; total power is
preserved.  This is intentionally a higher-upside / higher-risk submission.
"""

from __future__ import annotations

import argparse
import json
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
from powerfield_common import ASSETS, TINY, geom_effective_dist, regime_of


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


def mean_pair_cos(v: np.ndarray) -> float:
    x = v.reshape(v.shape[0], -1).astype(np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + TINY)
    g = x @ x.T
    iu = np.triu_indices(len(x), 1)
    return float(g[iu].mean()) if len(iu[0]) else 1.0


def predict_joint_targets(
    coll: np.ndarray,
    pos_train: np.ndarray,
    blocked_train: np.ndarray,
    pos_test: np.ndarray,
    blocked_test: np.ndarray,
    k: int,
    power: float,
    geom_mix: float,
    floor: float,
) -> dict[str, np.ndarray]:
    total = np.maximum(coll.sum(axis=(1, 2, 3)), TINY).astype(np.float64)
    n = len(pos_test)
    target = np.empty((n, MH, MV, S_SC), dtype=np.float32)
    agree = np.empty(n, dtype=np.float32)
    eff_nn = np.empty(n, dtype=np.float32)
    entropy = np.empty(n, dtype=np.float32)
    nn_weight = np.empty(n, dtype=np.float32)

    for i in range(n):
        eff = geom_effective_dist(pos_train, blocked_train, pos_test[i, :2], blocked_test[i]).copy()
        eff[total <= 0] = np.inf
        idx = np.argpartition(eff, k)[:k]
        idx = idx[np.argsort(eff[idx])]
        w = 1.0 / np.maximum(eff[idx], 0.30) ** power
        w = w / np.sum(w)
        nb = np.asarray(coll[idx], dtype=np.float32)
        nb = nb / np.maximum(total[idx, None, None, None].astype(np.float32), TINY)

        arith = np.sum(w[:, None, None, None].astype(np.float32) * nb, axis=0)
        log_mean = np.sum(w[:, None, None, None].astype(np.float32) * np.log(nb + floor), axis=0)
        geom = np.exp(log_mean).astype(np.float32)
        geom = geom / np.maximum(np.sum(geom), TINY)

        out = (1.0 - geom_mix) * arith + geom_mix * geom
        out = out / np.maximum(np.sum(out), TINY)
        target[i] = out.astype(np.float32)
        agree[i] = mean_pair_cos(nb)
        eff_nn[i] = float(eff[idx[0]])
        entropy[i] = float(-(w * np.log(w + 1e-12)).sum())
        nn_weight[i] = float(w[0])
        if (i + 1) % 50 == 0:
            print(f"joint targets {i+1}/{n}", flush=True)
    return {
        "joint": target,
        "agree": agree,
        "eff_nn": eff_nn,
        "entropy": entropy,
        "nn_weight": nn_weight,
    }


def marginal_stats_from_collapsed(coll: np.ndarray) -> dict[str, np.ndarray]:
    total = np.maximum(coll.sum(axis=(1, 2, 3)), TINY)
    beam = coll.sum(axis=-1) / total[:, None, None]
    delay = coll.sum(axis=(1, 2)) / total[:, None]
    return {"total": total, "beam": beam, "delay": delay}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=ASSETS + "/cache_v91/v91_Round1_Test_Channel.npy")
    parser.add_argument("--output", default=str(PATHS.cache_dir(92) / "v92_Round1_Test_Channel.npy"))
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    parser.add_argument("--coll", default=ASSETS + "/truth_coll_power.npy")
    parser.add_argument("--report", default=ASSETS + "/v92_joint_ad_spectral_matching_generate.json")
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--power", type=float, default=3.2)
    parser.add_argument("--geom-mix", type=float, default=0.45)
    parser.add_argument("--target-floor", type=float, default=1e-11)
    parser.add_argument("--eta", default="clear:0.10,light:0.14,medium:0.34,heavy:0.30")
    parser.add_argument("--agree-center", type=float, default=0.55)
    parser.add_argument("--agree-width", type=float, default=0.35)
    parser.add_argument("--min-gate", type=float, default=0.45)
    parser.add_argument("--max-gate", type=float, default=1.35)
    parser.add_argument("--gain-clip", default="0.58,1.82")
    parser.add_argument("--top-gain-clip", default="0.70,1.55")
    parser.add_argument("--top-components", type=int, default=128)
    parser.add_argument("--batch", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data)
    pos_train = np.load(data_dir / "Round1_Train_Pos.npy")
    pos_test = np.load(data_dir / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked_test)
    eta_base = parse_regime_values(args.eta, 0.0)
    gain_clip = tuple(float(x) for x in args.gain_clip.split(","))
    top_gain_clip = tuple(float(x) for x in args.top_gain_clip.split(","))

    coll = np.load(args.coll, mmap_mode="r")
    if coll.shape != (2000, MH, MV, S_SC):
        raise ValueError(f"unexpected coll shape {coll.shape}")
    targets = predict_joint_targets(
        coll,
        pos_train,
        blocked_train,
        pos_test,
        blocked_test,
        args.k,
        args.power,
        args.geom_mix,
        args.target_floor,
    )

    agree_gate = args.min_gate + (args.max_gate - args.min_gate) * np.clip(
        (targets["agree"] - args.agree_center) / max(args.agree_width, 1e-6),
        0.0,
        1.0,
    )
    eta = np.array([eta_base[str(r)] for r in regimes], dtype=np.float32) * agree_gate.astype(np.float32)

    base = np.load(args.base, mmap_mode="r")
    if base.shape != (500, 256, 4, 192) or base.dtype != np.complex64:
        raise ValueError(f"unexpected base shape/dtype {base.shape} {base.dtype}")

    beta_s = beta_subcarriers(args.beta, args.gamma)
    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)

    stats = {
        "power_ratio": [],
        "delta_nmse_to_base_num": 0.0,
        "delta_nmse_to_base_den": 0.0,
        "joint_cos_to_target_before": [],
        "joint_cos_to_target_after": [],
        "pas_cos_to_base": [],
        "pdp_cos_to_base": [],
        "gain_min": [],
        "gain_max": [],
        "eta": eta.tolist(),
    }
    by_regime = {r: {"count": 0, "eta": [], "joint_cos_before": [], "joint_cos_after": []} for r in REGIMES}

    for start in range(0, 500, args.batch):
        end = min(start + args.batch, 500)
        n = end - start
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        p0 = (np.abs(ad0) ** 2).astype(np.float32)
        coll0 = p0.sum(axis=(1, 4))
        total0 = np.maximum(coll0.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        cur = coll0 / total0[:, None, None, None]
        tgt = targets["joint"][start:end].astype(np.float32)

        exponent = eta[start:end].astype(np.float32)
        ratio_hvt = ((tgt + args.target_floor) / (cur + args.target_floor)) ** exponent[:, None, None, None]
        p = p0 * ratio_hvt[:, None, :, :, None, :]

        gain = np.sqrt(p / np.maximum(p0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, gain_clip[0], gain_clip[1])

        # Keep the strongest existing components more protected; v91/v87 likely
        # already place their phase reasonably there.
        flat_power = coll0.reshape(n, -1)
        ktop = min(args.top_components, flat_power.shape[1])
        top_ids = np.argpartition(flat_power, -ktop, axis=1)[:, -ktop:]
        top_mask = np.zeros_like(flat_power, dtype=bool)
        top_mask[np.arange(n)[:, None], top_ids] = True
        top_mask = top_mask.reshape(n, MH, MV, S_SC)[:, None, :, :, None, :]
        gain = np.where(top_mask, np.clip(gain, top_gain_clip[0], top_gain_clip[1]), gain)

        ad1 = (ad0 * gain).astype(np.complex64)
        p1 = (np.abs(ad1) ** 2).astype(np.float32)
        # Preserve original per-point total power after clipping.
        total1 = np.maximum(p1.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        ad1 *= np.sqrt(total0 / total1)[:, None, None, None, None, None].astype(np.float32)

        h1 = inverse_ad(ad1).reshape(n, 256, 4, 192)
        h1 = h1 * np.exp(-1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        out[start:end] = h1.astype(np.complex64)
        out.flush()

        p1 = (np.abs(ad1) ** 2).astype(np.float32)
        coll1 = p1.sum(axis=(1, 4))
        total_after = np.maximum(coll1.sum(axis=(1, 2, 3)), TINY)
        cur1 = coll1 / total_after[:, None, None, None]

        m0 = marginal_stats_from_collapsed(coll0)
        m1 = marginal_stats_from_collapsed(coll1)
        stats["power_ratio"].extend((total_after / total0).tolist())
        stats["joint_cos_to_target_before"].extend(row_cos(cur, tgt).tolist())
        stats["joint_cos_to_target_after"].extend(row_cos(cur1, tgt).tolist())
        stats["pas_cos_to_base"].extend(row_cos(m1["beam"], m0["beam"]).tolist())
        stats["pdp_cos_to_base"].extend(row_cos(m1["delay"], m0["delay"]).tolist())
        stats["gain_min"].extend(np.min(gain.reshape(n, -1), axis=1).tolist())
        stats["gain_max"].extend(np.max(gain.reshape(n, -1), axis=1).tolist())
        diff = h1 - h0
        stats["delta_nmse_to_base_num"] += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
        stats["delta_nmse_to_base_den"] += float(np.sum(np.abs(h0) ** 2, dtype=np.float64))
        before = row_cos(cur, tgt)
        after = row_cos(cur1, tgt)
        for j, reg in enumerate(regimes[start:end]):
            b = by_regime[str(reg)]
            b["count"] += 1
            b["eta"].append(float(exponent[j]))
            b["joint_cos_before"].append(float(before[j]))
            b["joint_cos_after"].append(float(after[j]))
        print(f"generated {end}/500", flush=True)

    payload = {
        "method": "v92_joint_ad_spectral_matching",
        "base": args.base,
        "operation": "joint PHV AD collapsed-power projection to sharp geometry-kNN target; phase inherited from v91; total power preserved",
        "neighbor": {"k": int(args.k), "power": float(args.power), "geom_mix": float(args.geom_mix)},
        "eta_base": eta_base,
        "agree_gate": {
            "center": float(args.agree_center),
            "width": float(args.agree_width),
            "min": float(args.min_gate),
            "max": float(args.max_gate),
        },
        "clips": {"gain": gain_clip, "top_gain": top_gain_clip, "top_components": int(args.top_components)},
        "target_agreement": {
            "joint": summarize(targets["agree"]),
            "eff_nn": summarize(targets["eff_nn"]),
            "entropy": summarize(targets["entropy"]),
            "nn_weight": summarize(targets["nn_weight"]),
        },
        "overall": {
            "power_ratio": summarize(stats["power_ratio"]),
            "eta": summarize(stats["eta"]),
            "joint_cos_to_target_before": summarize(stats["joint_cos_to_target_before"]),
            "joint_cos_to_target_after": summarize(stats["joint_cos_to_target_after"]),
            "pas_cos_to_base": summarize(stats["pas_cos_to_base"]),
            "pdp_cos_to_base": summarize(stats["pdp_cos_to_base"]),
            "gain_min": summarize(stats["gain_min"]),
            "gain_max": summarize(stats["gain_max"]),
            "delta_nmse_to_base": stats["delta_nmse_to_base_num"] / max(stats["delta_nmse_to_base_den"], TINY),
        },
        "by_regime": {r: {k: summarize(v) if isinstance(v, list) else v for k, v in vals.items()} for r, vals in by_regime.items()},
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
