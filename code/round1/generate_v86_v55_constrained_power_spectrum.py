#!/usr/bin/env python3
"""Generate v86: constrained power-spectrum calibration on top of v55.

Goal:
  Keep the hard part learned by v55 (phase, ADShift, top-component residual
  structure), and only correct easier non-coherent power-spectrum envelopes.

Operation:
  1. Start from the v55 complex submission output.
  2. De-rotate and transform to PHV angle-delay domain.
  3. Predict target beam/delay marginal shapes from sharp geometry-gated
     neighbors in the training true channels.
  4. Apply only positive real amplitude gains in AD domain.
  5. Protect strong existing components and preserve per-point total power.

This version intentionally does NOT:
  - change phase,
  - add new complex residuals,
  - move AD bins,
  - replace v55 top component support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import PATHS
from powerfield_common import ASSETS, TINY, build_collapsed_caches, regime_of, sharp_marginals


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


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return num / np.maximum(den, 1e-30)


def top_masks(power: np.ndarray, top4: int, top16: int) -> tuple[np.ndarray, np.ndarray]:
    flat = power.reshape(power.shape[0], -1)
    k16 = min(int(top16), flat.shape[1])
    k4 = min(int(top4), k16)
    ids16 = np.argpartition(flat, -k16, axis=1)[:, -k16:]
    mask16 = np.zeros_like(flat, dtype=bool)
    rows = np.arange(flat.shape[0])[:, None]
    mask16[rows, ids16] = True
    vals16 = np.take_along_axis(flat, ids16, axis=1)
    local4 = np.argpartition(vals16, -k4, axis=1)[:, -k4:]
    ids4 = np.take_along_axis(ids16, local4, axis=1)
    mask4 = np.zeros_like(flat, dtype=bool)
    mask4[rows, ids4] = True
    return mask4.reshape(power.shape), mask16.reshape(power.shape)


def summarize(x: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=ASSETS + "/v55_Round1_Test_Channel.npy")
    parser.add_argument("--output", default=str(PATHS.cache_dir(86) / "v86_Round1_Test_Channel.npy"))
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    parser.add_argument("--report", default=ASSETS + "/v86_constrained_power_spectrum_generate.json")
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--sharp-k", type=int, default=8)
    parser.add_argument("--sharp-p", type=float, default=3.0)
    parser.add_argument(
        "--eta-beam",
        default="clear:0.0,light:0.0,medium:0.12,heavy:0.10",
        help="regime etas for beam marginal shape",
    )
    parser.add_argument(
        "--eta-delay",
        default="clear:0.0,light:0.0,medium:0.08,heavy:0.04",
        help="regime etas for delay marginal shape",
    )
    parser.add_argument("--top4-components", type=int, default=4)
    parser.add_argument("--top16-components", type=int, default=16)
    parser.add_argument("--top4-clip", default="0.98,1.035")
    parser.add_argument("--top16-clip", default="0.95,1.08")
    parser.add_argument("--rest-clip", default="0.85,1.15")
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--preserve-total", action="store_true", default=True)
    parser.add_argument("--batch", type=int, default=25)
    args = parser.parse_args()

    eta_beam = parse_regime_values(args.eta_beam, 0.0)
    eta_delay = parse_regime_values(args.eta_delay, 0.0)
    top4_lo, top4_hi = [float(x) for x in args.top4_clip.split(",")]
    top16_lo, top16_hi = [float(x) for x in args.top16_clip.split(",")]
    rest_lo, rest_hi = [float(x) for x in args.rest_clip.split(",")]

    pos_train = np.load(Path(args.data) / "Round1_Train_Pos.npy")
    pos_test = np.load(Path(args.data) / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked_test)

    coll, _base_coll = build_collapsed_caches()
    total_train = coll.sum((1, 2, 3))
    bnorm_train = coll.sum(-1).reshape(len(coll), 128) / np.maximum(total_train[:, None], TINY)
    dnorm_train = coll.sum((1, 2)) / np.maximum(total_train[:, None], TINY)

    print("predicting sharp marginal shapes ...", flush=True)
    target_beam = np.empty((500, 128), dtype=np.float32)
    target_delay = np.empty((500, 192), dtype=np.float32)
    agree_beam = np.empty(500, dtype=np.float32)
    agree_delay = np.empty(500, dtype=np.float32)
    eff_nn = np.empty(500, dtype=np.float32)
    for i in range(500):
        sb, sd, ab, ad, _std_t, eff = sharp_marginals(
            bnorm_train,
            dnorm_train,
            total_train,
            pos_train,
            blocked_train,
            pos_test[i, :2],
            blocked_test[i],
            k=args.sharp_k,
            p=args.sharp_p,
        )
        target_beam[i] = sb
        target_delay[i] = sd
        agree_beam[i] = ab
        agree_delay[i] = ad
        eff_nn[i] = eff[0]

    base = np.load(args.base, mmap_mode="r")
    if base.shape != (500, 256, 4, 192) or base.dtype != np.complex64:
        raise ValueError(f"unexpected base shape/dtype {base.shape} {base.dtype}")

    beta_s = beta_subcarriers(args.beta, args.gamma)
    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)

    stats = {
        "gain_min": [],
        "gain_max": [],
        "power_ratio": [],
        "pdp_cos_to_base": [],
        "pas_cos_to_base": [],
        "top4_frac_delta": [],
        "top16_frac_delta": [],
    }
    by_regime = {r: {"power_ratio": [], "pdp_cos_to_base": [], "pas_cos_to_base": [], "top4_frac_delta": []}
                 for r in ["clear", "light", "medium", "heavy"]}

    for start in range(0, 500, args.batch):
        end = min(start + args.batch, 500)
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        p0 = (np.abs(ad0) ** 2).sum(axis=(1, 4)).astype(np.float64)  # B,H,V,tau
        total0 = np.maximum(p0.sum(axis=(1, 2, 3), keepdims=True), TINY)
        base_beam = p0.sum(axis=-1, keepdims=True) / total0
        base_delay = p0.sum(axis=(1, 2), keepdims=True) / total0

        bp = target_beam[start:end].reshape(end - start, 16, 8, 1)
        dp = target_delay[start:end].reshape(end - start, 1, 1, 192)
        eb = np.array([eta_beam[str(r)] for r in regimes[start:end]], dtype=np.float32).reshape(-1, 1, 1, 1)
        ed = np.array([eta_delay[str(r)] for r in regimes[start:end]], dtype=np.float32).reshape(-1, 1, 1, 1)

        delta_b = float(args.delta) / 128.0
        delta_d = float(args.delta) / 192.0
        # Positive real amplitude gain.  Power-shape ratio uses eta/2 because
        # multiplying complex coefficients by gain squares the power.
        gain_beam = ((bp + delta_b) / (base_beam + delta_b)) ** (0.5 * eb)
        gain_delay = ((dp + delta_d) / (base_delay + delta_d)) ** (0.5 * ed)
        gain_hvt = (gain_beam * gain_delay).astype(np.float32)

        mask4, mask16 = top_masks(p0, args.top4_components, args.top16_components)
        gain_hvt = np.clip(gain_hvt, rest_lo, rest_hi)
        gain_hvt = np.where(mask16, np.clip(gain_hvt, top16_lo, top16_hi), gain_hvt)
        gain_hvt = np.where(mask4, np.clip(gain_hvt, top4_lo, top4_hi), gain_hvt)

        ad1 = (ad0 * gain_hvt[:, None, :, :, None, :].astype(np.float32)).astype(np.complex64)
        if args.preserve_total:
            p1_total = np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64)
            p0_total = np.sum(np.abs(ad0) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64)
            scale = np.sqrt(p0_total / np.maximum(p1_total, 1e-30)).astype(np.float32)
            ad1 *= scale[:, None, None, None, None, None]

        h1 = inverse_ad(ad1).reshape(end - start, 256, 4, 192)
        h1 = h1 * np.exp(-1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        out[start:end] = h1.astype(np.complex64)
        out.flush()

        p1 = (np.abs(ad1) ** 2).sum(axis=(1, 4)).astype(np.float64)
        total1 = np.maximum(p1.sum(axis=(1, 2, 3)), TINY)
        total0v = np.maximum(p0.sum(axis=(1, 2, 3)), TINY)
        beam0 = p0.sum(axis=-1).reshape(end - start, -1)
        beam1 = p1.sum(axis=-1).reshape(end - start, -1)
        delay0 = p0.sum(axis=(1, 2))
        delay1 = p1.sum(axis=(1, 2))
        flat0 = p0.reshape(end - start, -1)
        flat1 = p1.reshape(end - start, -1)
        top4_0 = np.sort(flat0, axis=1)[:, -args.top4_components:].sum(axis=1) / total0v
        top4_1 = np.sort(flat1, axis=1)[:, -args.top4_components:].sum(axis=1) / total1
        top16_0 = np.sort(flat0, axis=1)[:, -args.top16_components:].sum(axis=1) / total0v
        top16_1 = np.sort(flat1, axis=1)[:, -args.top16_components:].sum(axis=1) / total1
        pr = total1 / total0v
        pdpc = row_cos(delay1, delay0)
        pasc = row_cos(beam1, beam0)
        d4 = top4_1 - top4_0
        d16 = top16_1 - top16_0
        stats["gain_min"].extend(np.min(gain_hvt.reshape(end - start, -1), axis=1).tolist())
        stats["gain_max"].extend(np.max(gain_hvt.reshape(end - start, -1), axis=1).tolist())
        stats["power_ratio"].extend(pr.tolist())
        stats["pdp_cos_to_base"].extend(pdpc.tolist())
        stats["pas_cos_to_base"].extend(pasc.tolist())
        stats["top4_frac_delta"].extend(d4.tolist())
        stats["top16_frac_delta"].extend(d16.tolist())
        for j, reg in enumerate(regimes[start:end]):
            b = by_regime[str(reg)]
            b["power_ratio"].append(float(pr[j]))
            b["pdp_cos_to_base"].append(float(pdpc[j]))
            b["pas_cos_to_base"].append(float(pasc[j]))
            b["top4_frac_delta"].append(float(d4[j]))
        print(f"generated {end}/500", flush=True)

    payload = {
        "method": "v86_v55_constrained_power_spectrum",
        "base": args.base,
        "operation": "PHV AD-domain positive amplitude gain only; phase/support frozen; per-point power preserved",
        "eta_beam": eta_beam,
        "eta_delay": eta_delay,
        "sharp": {"k": int(args.sharp_k), "p": float(args.sharp_p)},
        "clips": {
            "top4": [top4_lo, top4_hi],
            "top16": [top16_lo, top16_hi],
            "rest": [rest_lo, rest_hi],
        },
        "preserve_total": bool(args.preserve_total),
        "regime_counts": {r: int(np.sum(regimes == r)) for r in ["clear", "light", "medium", "heavy"]},
        "agreement": {
            "beam": summarize(agree_beam),
            "delay": summarize(agree_delay),
            "eff_nn": summarize(eff_nn),
        },
        "overall": {k: summarize(v) for k, v in stats.items()},
        "by_regime": {r: {k: summarize(v) for k, v in vals.items()} for r, vals in by_regime.items()},
        "output": args.output,
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
