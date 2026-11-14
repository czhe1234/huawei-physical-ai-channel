#!/usr/bin/env python3
"""Generate v87: easy-field projection on top of the current hard baseline.

This version keeps the hard part from the current best-like submission:
phase, strong AD component support, and coherent residual structure.  It
only projects the output toward easy-to-predict non-coherent targets:

  - total power, with a small per-regime exponent,
  - PAS / beam-power shape,
  - PDP / delay-power shape,
  - BS polarization energy split,
  - UE antenna energy split,
  - top AD component energy concentration.

All corrections are positive real amplitude gains in PHV angle-delay domain.
No phase is learned or changed directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import PATHS
from powerfield_common import ASSETS, TINY, geom_effective_dist, regime_of


MP, MH, MV, N_ANT, S_SC = 2, 16, 8, 4, 192
BS_POS = np.array([50.0, 0.0, 25.0], dtype=np.float64)


def parse_regime_values(text: str, default: float) -> dict[str, float]:
    out = {"clear": default, "light": default, "medium": default, "heavy": default}
    if not text:
        return out
    for item in text.split(","):
        if not item:
            continue
        key, val = item.split(":")
        key = key.strip()
        if key not in out:
            raise ValueError(f"unknown regime {key!r}")
        out[key] = float(val)
    return out


def beta_subcarriers(beta: float, gamma: float) -> np.ndarray:
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    return (float(beta) + float(gamma) * sub).astype(np.float32)


def ad_transform(h: np.ndarray) -> np.ndarray:
    cir = np.fft.ifft(h, axis=-1)
    arr = cir.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    return np.fft.fft2(arr, axes=(2, 3)).astype(np.complex64)


def inverse_ad(ad: np.ndarray) -> np.ndarray:
    arr = np.fft.ifft2(ad, axes=(2, 3))
    cir = arr.reshape(ad.shape[0], MP * MH * MV, N_ANT, S_SC)
    return np.fft.fft(cir, axis=-1).astype(np.complex64)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(x.sum(axis=tuple(range(1, x.ndim)), keepdims=True), TINY)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return num / np.maximum(den, 1e-30)


def summarize(values: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "mean": float(np.mean(a)),
        "p10": float(np.quantile(a, 0.10)),
        "median": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def top_masks_from_collapsed(power_hvt: np.ndarray, top4: int, top16: int) -> tuple[np.ndarray, np.ndarray]:
    flat = power_hvt.reshape(power_hvt.shape[0], -1)
    k16 = min(int(top16), flat.shape[1])
    k4 = min(int(top4), k16)
    ids16 = np.argpartition(flat, -k16, axis=1)[:, -k16:]
    rows = np.arange(flat.shape[0])[:, None]
    mask16 = np.zeros_like(flat, dtype=bool)
    mask16[rows, ids16] = True
    vals16 = np.take_along_axis(flat, ids16, axis=1)
    local4 = np.argpartition(vals16, -k4, axis=1)[:, -k4:]
    ids4 = np.take_along_axis(ids16, local4, axis=1)
    mask4 = np.zeros_like(flat, dtype=bool)
    mask4[rows, ids4] = True
    return mask4.reshape(power_hvt.shape), mask16.reshape(power_hvt.shape)


def build_easy_cache(cache_path: Path, data_dir: Path, beta_s: np.ndarray, batch: int) -> dict[str, np.ndarray]:
    if cache_path.exists():
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    pos_train = np.load(data_dir / "Round1_Train_Pos.npy")
    train = np.load(data_dir / "Round1_Train_Channel.npy", mmap_mode="r")
    d_train = np.linalg.norm(pos_train - BS_POS[None, :], axis=1)
    n = train.shape[0]

    total = np.empty(n, dtype=np.float32)
    beam = np.empty((n, MH * MV), dtype=np.float32)
    delay = np.empty((n, S_SC), dtype=np.float32)
    pol = np.empty((n, MP), dtype=np.float32)
    ue = np.empty((n, N_ANT), dtype=np.float32)
    top4 = np.empty(n, dtype=np.float32)
    top16 = np.empty(n, dtype=np.float32)

    print("building v87 train easy cache ...", flush=True)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        h = np.asarray(train[s:e], dtype=np.complex64)
        h = h * np.exp(1j * d_train[s:e, None, None, None] * beta_s).astype(np.complex64)
        ad = ad_transform(h)
        p = (np.abs(ad) ** 2).astype(np.float32)  # B,2,16,8,4,192
        coll = p.sum(axis=(1, 4))  # B,16,8,192
        t = coll.sum(axis=(1, 2, 3))
        total[s:e] = t
        beam[s:e] = coll.sum(axis=-1).reshape(e - s, -1) / np.maximum(t[:, None], TINY)
        delay[s:e] = coll.sum(axis=(1, 2)) / np.maximum(t[:, None], TINY)
        pol[s:e] = p.sum(axis=(2, 3, 4, 5)) / np.maximum(t[:, None], TINY)
        ue[s:e] = p.sum(axis=(1, 2, 3, 5)) / np.maximum(t[:, None], TINY)
        flat = coll.reshape(e - s, -1)
        sorted_tail = np.sort(flat, axis=1)
        top4[s:e] = sorted_tail[:, -4:].sum(axis=1) / np.maximum(t, TINY)
        top16[s:e] = sorted_tail[:, -16:].sum(axis=1) / np.maximum(t, TINY)
        print(f"cached train {e}/{n}", flush=True)

    np.savez_compressed(
        cache_path,
        total=total,
        log_total=np.log(total + TINY).astype(np.float32),
        beam=beam.astype(np.float32),
        delay=delay.astype(np.float32),
        pol=pol.astype(np.float32),
        ue=ue.astype(np.float32),
        top4=top4.astype(np.float32),
        top16=top16.astype(np.float32),
    )
    data = np.load(cache_path)
    return {k: data[k] for k in data.files}


def mean_pair_cos(v: np.ndarray) -> float:
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + TINY)
    g = vn @ vn.T
    iu = np.triu_indices(len(v), 1)
    return float(g[iu].mean()) if len(iu[0]) else 1.0


def predict_easy_targets(
    bank: dict[str, np.ndarray],
    pos_train: np.ndarray,
    blocked_train: np.ndarray,
    pos_test: np.ndarray,
    blocked_test: np.ndarray,
    k: int,
    p: float,
) -> dict[str, np.ndarray]:
    n = len(pos_test)
    out = {
        "total": np.empty(n, dtype=np.float32),
        "beam": np.empty((n, MH * MV), dtype=np.float32),
        "delay": np.empty((n, S_SC), dtype=np.float32),
        "pol": np.empty((n, MP), dtype=np.float32),
        "ue": np.empty((n, N_ANT), dtype=np.float32),
        "top4": np.empty(n, dtype=np.float32),
        "top16": np.empty(n, dtype=np.float32),
        "agree_beam": np.empty(n, dtype=np.float32),
        "agree_delay": np.empty(n, dtype=np.float32),
        "eff_nn": np.empty(n, dtype=np.float32),
    }
    for i in range(n):
        eff = geom_effective_dist(pos_train, blocked_train, pos_test[i, :2], blocked_test[i]).copy()
        eff[bank["total"] <= 0] = np.inf
        idx = np.argpartition(eff, k)[:k]
        idx = idx[np.argsort(eff[idx])]
        w = 1.0 / np.maximum(eff[idx], 0.3) ** p
        w = w / np.sum(w)
        out["total"][i] = float(np.exp(np.sum(w * bank["log_total"][idx])))
        for key in ["beam", "delay", "pol", "ue"]:
            val = np.sum(w.reshape((-1,) + (1,) * (bank[key].ndim - 1)) * bank[key][idx], axis=0)
            val = val / np.maximum(np.sum(val), TINY)
            out[key][i] = val.astype(np.float32)
        out["top4"][i] = float(np.sum(w * bank["top4"][idx]))
        out["top16"][i] = float(np.sum(w * bank["top16"][idx]))
        out["agree_beam"][i] = mean_pair_cos(bank["beam"][idx])
        out["agree_delay"][i] = mean_pair_cos(bank["delay"][idx])
        out["eff_nn"][i] = float(eff[idx[0]])
    return out


def marginal_stats_from_power(p: np.ndarray) -> dict[str, np.ndarray]:
    total = np.maximum(p.sum(axis=(1, 2, 3, 4, 5)), TINY)
    beam = p.sum(axis=(1, 4, 5)) / total[:, None, None]
    delay = p.sum(axis=(1, 2, 3, 4)) / total[:, None]
    pol = p.sum(axis=(2, 3, 4, 5)) / total[:, None]
    ue = p.sum(axis=(1, 2, 3, 5)) / total[:, None]
    return {"total": total, "beam": beam, "delay": delay, "pol": pol, "ue": ue}


def apply_ratio(power: np.ndarray, target: np.ndarray, current: np.ndarray,
                exponent: np.ndarray, axes: tuple[int, ...], delta: float) -> np.ndarray:
    """Apply multiplicative power ratio broadcast over the complementary axes."""
    ratio = ((target + delta) / (current + delta)) ** exponent
    shape = [power.shape[0], 1, 1, 1, 1, 1]
    for ax, size in zip(axes, target.shape[1:]):
        shape[ax] = size
    return power * ratio.reshape(shape).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=ASSETS + "/cache_v83/v83_Round1_Test_Channel.npy")
    parser.add_argument("--output", default=str(PATHS.cache_dir(87) / "v87_Round1_Test_Channel.npy"))
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    parser.add_argument("--cache", default=ASSETS + "/train_easy_marginals_v87_g010.npz")
    parser.add_argument("--report", default=ASSETS + "/v87_easy_field_projection_generate.json")
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--power", type=float, default=3.0)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--eta-beam", default="clear:0.000,light:0.000,medium:0.16,heavy:0.14")
    parser.add_argument("--eta-delay", default="clear:0.000,light:0.000,medium:0.10,heavy:0.06")
    parser.add_argument("--eta-pol", default="clear:0.000,light:0.000,medium:0.08,heavy:0.08")
    parser.add_argument("--eta-ue", default="clear:0.000,light:0.000,medium:0.08,heavy:0.08")
    parser.add_argument("--eta-total", default="clear:0.000,light:0.000,medium:0.000,heavy:0.000")
    parser.add_argument("--eta-top", default="clear:0.000,light:0.000,medium:0.12,heavy:0.10")
    parser.add_argument("--gain-clip-rest", default="0.75,1.25")
    parser.add_argument("--gain-clip-top16", default="0.92,1.12")
    parser.add_argument("--gain-clip-top4", default="0.97,1.05")
    parser.add_argument("--total-clip", default="0.85,1.25")
    parser.add_argument("--top-delta-clip", type=float, default=0.018)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()

    eta = {
        "beam": parse_regime_values(args.eta_beam, 0.0),
        "delay": parse_regime_values(args.eta_delay, 0.0),
        "pol": parse_regime_values(args.eta_pol, 0.0),
        "ue": parse_regime_values(args.eta_ue, 0.0),
        "total": parse_regime_values(args.eta_total, 0.0),
        "top": parse_regime_values(args.eta_top, 0.0),
    }
    clip_rest = tuple(float(x) for x in args.gain_clip_rest.split(","))
    clip_top16 = tuple(float(x) for x in args.gain_clip_top16.split(","))
    clip_top4 = tuple(float(x) for x in args.gain_clip_top4.split(","))
    total_clip = tuple(float(x) for x in args.total_clip.split(","))

    data_dir = Path(args.data)
    pos_train = np.load(data_dir / "Round1_Train_Pos.npy")
    pos_test = np.load(data_dir / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked_test)

    beta_s = beta_subcarriers(args.beta, args.gamma)
    bank = build_easy_cache(Path(args.cache), data_dir, beta_s, args.batch)
    print("predicting v87 easy targets ...", flush=True)
    targets = predict_easy_targets(
        bank, pos_train, blocked_train, pos_test, blocked_test, args.k, args.power
    )

    base = np.load(args.base, mmap_mode="r")
    if base.shape != (500, 256, 4, 192) or base.dtype != np.complex64:
        raise ValueError(f"unexpected base shape/dtype {base.shape} {base.dtype}")

    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)

    stats = {
        "power_ratio": [],
        "target_total_ratio": [],
        "pdp_cos_to_base": [],
        "pas_cos_to_base": [],
        "pol_cos_to_base": [],
        "ue_cos_to_base": [],
        "gain_min": [],
        "gain_max": [],
        "top4_frac_delta": [],
        "top16_frac_delta": [],
    }

    by_regime = {
        r: {"power_ratio": [], "pdp_cos_to_base": [], "pas_cos_to_base": [], "top4_frac_delta": []}
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

        et = {}
        for name in eta:
            arr = np.array([eta[name][str(r)] for r in regimes[start:end]], dtype=np.float32)
            et[name] = arr

        target = {
            "beam": targets["beam"][start:end].reshape(n, MH, MV).astype(np.float32),
            "delay": targets["delay"][start:end].astype(np.float32),
            "pol": targets["pol"][start:end].astype(np.float32),
            "ue": targets["ue"][start:end].astype(np.float32),
        }

        for _ in range(args.iters):
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["beam"], cur["beam"], (et["beam"] / args.iters)[:, None, None], (2, 3), 1e-8 / (MH * MV))
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["delay"], cur["delay"], (et["delay"] / args.iters)[:, None], (5,), 1e-8 / S_SC)
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["pol"], cur["pol"], (et["pol"] / args.iters)[:, None], (1,), 1e-8 / MP)
            cur = marginal_stats_from_power(p)
            p = apply_ratio(p, target["ue"], cur["ue"], (et["ue"] / args.iters)[:, None], (4,), 1e-8 / N_ANT)

        p0_coll = p0.sum(axis=(1, 4))
        p_coll = p.sum(axis=(1, 4))
        mask4, mask16 = top_masks_from_collapsed(p0_coll, 4, 16)
        flat_coll = p_coll.reshape(n, -1)
        flat_mask4 = mask4.reshape(n, -1)
        cur_total = np.maximum(flat_coll.sum(axis=1), TINY)
        cur4 = np.sum(flat_coll * flat_mask4, axis=1) / cur_total
        target4_raw = targets["top4"][start:end].astype(np.float32)
        desired4 = cur4 * np.maximum(target4_raw / np.maximum(cur4, TINY), TINY) ** et["top"]
        desired4 = np.clip(desired4, cur4 - args.top_delta_clip, cur4 + args.top_delta_clip)
        desired4 = np.clip(desired4, 1e-4, 0.98)
        top_scale = desired4 / np.maximum(cur4, TINY)
        rest_scale = (1.0 - desired4) / np.maximum(1.0 - cur4, TINY)
        scale_hvt = np.where(mask4, top_scale[:, None, None, None], rest_scale[:, None, None, None])
        p *= scale_hvt[:, None, :, :, None, :].astype(np.float32)

        p0_total = np.maximum(p0.sum(axis=(1, 2, 3, 4, 5)), TINY)
        target_total_ratio = targets["total"][start:end] / p0_total
        total_ratio = target_total_ratio ** et["total"]
        total_ratio = np.clip(total_ratio, total_clip[0], total_clip[1])
        p_total = np.maximum(p.sum(axis=(1, 2, 3, 4, 5)), TINY)
        p *= (p0_total * total_ratio / p_total)[:, None, None, None, None, None].astype(np.float32)

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
        flat0 = p0_coll.reshape(n, -1)
        total1 = np.maximum(flat1.sum(axis=1), TINY)
        total0 = np.maximum(flat0.sum(axis=1), TINY)
        top4_0 = np.sum(flat0 * flat_mask4, axis=1) / total0
        top4_1 = np.sum(flat1 * flat_mask4, axis=1) / total1
        mask16_flat = mask16.reshape(n, -1)
        top16_0 = np.sum(flat0 * mask16_flat, axis=1) / total0
        top16_1 = np.sum(flat1 * mask16_flat, axis=1) / total1

        stats["power_ratio"].extend((m1["total"] / m0["total"]).tolist())
        stats["target_total_ratio"].extend(target_total_ratio.tolist())
        stats["pdp_cos_to_base"].extend(row_cos(m1["delay"], m0["delay"]).tolist())
        stats["pas_cos_to_base"].extend(row_cos(m1["beam"], m0["beam"]).tolist())
        stats["pol_cos_to_base"].extend(row_cos(m1["pol"], m0["pol"]).tolist())
        stats["ue_cos_to_base"].extend(row_cos(m1["ue"], m0["ue"]).tolist())
        stats["gain_min"].extend(np.min(gain.reshape(n, -1), axis=1).tolist())
        stats["gain_max"].extend(np.max(gain.reshape(n, -1), axis=1).tolist())
        stats["top4_frac_delta"].extend((top4_1 - top4_0).tolist())
        stats["top16_frac_delta"].extend((top16_1 - top16_0).tolist())
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
        print(f"generated {end}/500", flush=True)

    payload = {
        "method": "v87_easy_field_projection",
        "base": args.base,
        "operation": "PHV AD positive amplitude projection to predicted easy marginals; phase frozen",
        "neighbor": {"k": int(args.k), "power": float(args.power)},
        "iters": int(args.iters),
        "eta": eta,
        "clips": {
            "rest": clip_rest,
            "top16": clip_top16,
            "top4": clip_top4,
            "total": total_clip,
            "top_delta": float(args.top_delta_clip),
        },
        "regime_counts": {r: int(np.sum(regimes == r)) for r in ["clear", "light", "medium", "heavy"]},
        "target_agreement": {
            "beam": summarize(targets["agree_beam"]),
            "delay": summarize(targets["agree_delay"]),
            "eff_nn": summarize(targets["eff_nn"]),
        },
        "overall": {k: summarize(v) for k, v in stats.items()},
        "by_regime": {r: {k: summarize(v) for k, v in vals.items()} for r, vals in by_regime.items()},
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
