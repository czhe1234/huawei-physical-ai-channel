#!/usr/bin/env python3
"""Decompose the remaining channel-reconstruction headroom with LOO oracles.

The current online line preserves phase, total power, and per-pol/per-UE
structure while modifying collapsed H/V/delay support.  This diagnostic uses
the strongest available train LOO full-AD cache and asks what would happen if
one factor at a time were known perfectly:

* point total power;
* collapsed H/V/delay support shape;
* full per-pol/per-UE AD magnitude;
* common phase, point complex scalar, block scalar, or H/V/delay component
  scalar;
* full component phase.

It is an oracle diagnostic only.  No test truth is used and no submission is
generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from power_calibration_lab import metrics_batch
from powerfield_common import ASSETS, TINY, regime_of


MP = 2
MH = 16
MV = 8
N_ANT = 4
S_SC = 192


def summarize(x: np.ndarray) -> dict[str, float]:
    v = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.nanmean(v)),
        "p10": float(np.nanquantile(v, 0.10)),
        "median": float(np.nanquantile(v, 0.50)),
        "p90": float(np.nanquantile(v, 0.90)),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
    }


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = a.reshape(a.shape[0], -1).to(torch.float64)
    bb = b.reshape(b.shape[0], -1).to(torch.float64)
    num = torch.sum(aa * bb, dim=1)
    den = torch.sqrt(torch.sum(aa * aa, dim=1) * torch.sum(bb * bb, dim=1)).clamp_min(1e-30)
    return num / den


def extra_metrics(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, np.ndarray]:
    pp = pred.real.square() + pred.imag.square()
    tp = truth.real.square() + truth.imag.square()
    pc = pp.sum(dim=(1, 4))
    tc = tp.sum(dim=(1, 4))
    beam = cosine_rows(pc.sum(dim=3), tc.sum(dim=3))
    delay = cosine_rows(pc.sum(dim=(1, 2)), tc.sum(dim=(1, 2)))
    joint = cosine_rows(pc, tc)
    full = cosine_rows(pp, tp)

    pb = pp.permute(0, 1, 4, 2, 3, 5).reshape(-1, MH * MV * S_SC)
    tb = tp.permute(0, 1, 4, 2, 3, 5).reshape(-1, MH * MV * S_SC)
    block = cosine_rows(pb, tb).reshape(pred.shape[0], MP * N_ANT).mean(dim=1)

    inner = torch.sum((torch.conj(pred) * truth).to(torch.complex128), dim=(1, 2, 3, 4, 5))
    pn = torch.sum(pred.abs().square().to(torch.float64), dim=(1, 2, 3, 4, 5))
    tn = torch.sum(truth.abs().square().to(torch.float64), dim=(1, 2, 3, 4, 5))
    coherence = inner.abs() / torch.sqrt(pn * tn).clamp_min(1e-30)
    power_ratio = pn / tn.clamp_min(1e-30)
    return {
        "joint_hvt": joint.detach().cpu().numpy(),
        "beam_hv": beam.detach().cpu().numpy(),
        "delay_t": delay.detach().cpu().numpy(),
        "full_power": full.detach().cpu().numpy(),
        "block_polue": block.detach().cpu().numpy(),
        "complex_coherence": coherence.detach().cpu().numpy(),
        "power_ratio": power_ratio.detach().cpu().numpy(),
    }


def rescale_energy(x: torch.Tensor, target_energy: torch.Tensor) -> torch.Tensor:
    energy = torch.sum(x.abs().square(), dim=(1, 2, 3, 4, 5)).clamp_min(1e-30)
    scale = torch.sqrt(target_energy / energy)
    return x * scale[:, None, None, None, None, None]


def make_variants(base: torch.Tensor, truth: torch.Tensor) -> dict[str, torch.Tensor]:
    eps = 1e-30
    bp = base.abs().square()
    tp = truth.abs().square()
    be = bp.sum(dim=(1, 2, 3, 4, 5)).clamp_min(eps)
    te = tp.sum(dim=(1, 2, 3, 4, 5)).clamp_min(eps)
    bc = bp.sum(dim=(1, 4))
    tc = tp.sum(dim=(1, 4))
    bcn = bc / be[:, None, None, None]
    tcn = tc / te[:, None, None, None]

    coll_shape_gain = torch.sqrt((tcn + eps) / (bcn + eps))
    coll_shape = base * coll_shape_gain[:, None, :, :, None, :]
    coll_shape = rescale_energy(coll_shape, be)

    coll_absolute_gain = torch.sqrt((tc + eps) / (bc + eps))
    coll_shape_total = base * coll_absolute_gain[:, None, :, :, None, :]
    coll_shape_total = rescale_energy(coll_shape_total, te)

    base_phase = torch.angle(base)
    truth_phase = torch.angle(truth)
    full_mag_shape = torch.polar(truth.abs(), base_phase)
    full_mag_shape = rescale_energy(full_mag_shape, be)
    full_magnitude = torch.polar(truth.abs(), base_phase)

    phase_only = torch.polar(base.abs(), truth_phase)
    phase_total = rescale_energy(phase_only, te)

    point_inner = torch.sum(torch.conj(base) * truth, dim=(1, 2, 3, 4, 5))
    point_den = torch.sum(base.abs().square(), dim=(1, 2, 3, 4, 5)).clamp_min(eps)
    point_scalar = point_inner / point_den
    point_complex_scalar = base * point_scalar[:, None, None, None, None, None]
    point_common_phase = base * torch.exp(1j * torch.angle(point_inner))[:, None, None, None, None, None]

    block_inner = torch.sum(torch.conj(base) * truth, dim=(2, 3, 5))
    block_den = torch.sum(base.abs().square(), dim=(2, 3, 5)).clamp_min(eps)
    block_scalar = block_inner / block_den
    block_complex_scalar = base * block_scalar[:, :, None, None, :, None]

    component_inner = torch.sum(torch.conj(base) * truth, dim=(1, 4))
    component_den = torch.sum(base.abs().square(), dim=(1, 4)).clamp_min(eps)
    component_scalar = component_inner / component_den
    component_complex_scalar = base * component_scalar[:, None, :, :, None, :]

    return {
        "base": base,
        "oracle_total_power": rescale_energy(base, te),
        "oracle_collapsed_shape": coll_shape,
        "oracle_collapsed_shape_total": coll_shape_total,
        "oracle_full_magnitude_shape": full_mag_shape,
        "oracle_full_magnitude": full_magnitude,
        "oracle_common_phase": point_common_phase,
        "oracle_point_complex_scalar": point_complex_scalar,
        "oracle_polue_block_scalar": block_complex_scalar,
        "oracle_hvt_component_scalar": component_complex_scalar,
        "oracle_full_phase": phase_only,
        "oracle_full_phase_total": phase_total,
        "truth": truth,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ad", default=ASSETS + "/train_v98like_loo_base_ad.npy")
    ap.add_argument("--residual-ad", default=ASSETS + "/train_v98like_loo_residual_ad.npy")
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--selection", default=ASSETS + "/feature_matched_candidates_500.json")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=ASSETS + "/v124_bottleneck_oracles_500.json")
    ap.add_argument("--out-md", default=ASSETS + "/v124_bottleneck_oracles_500.md")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    base_mm = np.load(args.base_ad, mmap_mode="r")
    residual_mm = np.load(args.residual_ad, mmap_mode="r")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    indices = np.asarray(selection["val_indices"][: int(args.count)], dtype=np.int64)
    blocked = np.load(args.geometry_cache)["train_blocked"]
    regimes = regime_of(blocked)[indices]

    accum: dict[str, dict[str, list[np.ndarray] | float]] = {}
    for start in range(0, len(indices), int(args.batch)):
        end = min(start + int(args.batch), len(indices))
        ids = indices[start:end]
        base = torch.from_numpy(np.asarray(base_mm[ids], dtype=np.complex64)).to(device)
        truth = base + torch.from_numpy(np.asarray(residual_mm[ids], dtype=np.complex64)).to(device)
        variants = make_variants(base, truth)
        for name, pred in variants.items():
            if name not in accum:
                accum[name] = {k: [] for k in [
                    "pas", "pdp", "e2", "t2", "p2", "dot", "joint_hvt", "beam_hv",
                    "delay_t", "full_power", "block_polue", "complex_coherence", "power_ratio"
                ]}
            official = metrics_batch(pred, truth)
            extra = extra_metrics(pred, truth)
            for key, value in official.items():
                accum[name][key].append(value)
            for key, value in extra.items():
                accum[name][key].append(value)
        del variants, base, truth
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"oracle {end}/{len(indices)}", flush=True)

    results: dict[str, dict[str, object]] = {}
    for name, raw in accum.items():
        vals = {key: np.concatenate(value) for key, value in raw.items()}
        nmse_global = float(vals["e2"].sum() / max(vals["t2"].sum(), TINY))
        nmse_point = vals["e2"] / np.maximum(vals["t2"], TINY)
        pas = float(np.mean(vals["pas"]))
        pdp = float(np.mean(vals["pdp"]))
        score = float(0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse_global))
        row: dict[str, object] = {
            "score_proxy": score,
            "pas": pas,
            "pdp": pdp,
            "nmse_global": nmse_global,
            "nmse_point": summarize(nmse_point),
        }
        for key in [
            "joint_hvt", "beam_hv", "delay_t", "full_power", "block_polue",
            "complex_coherence", "power_ratio"
        ]:
            row[key] = summarize(vals[key])
        for reg in ("clear", "light", "medium", "heavy"):
            mask = regimes == reg
            row[f"{reg}_count"] = int(mask.sum())
            row[f"{reg}_pas"] = float(np.mean(vals["pas"][mask]))
            row[f"{reg}_pdp"] = float(np.mean(vals["pdp"][mask]))
            row[f"{reg}_nmse"] = float(vals["e2"][mask].sum() / max(vals["t2"][mask].sum(), TINY))
        results[name] = row

    base_score = float(results["base"]["score_proxy"])
    for row in results.values():
        row["score_gain_vs_base"] = float(row["score_proxy"] - base_score)

    payload = {
        "method": "v124_bottleneck_oracle_decomposition",
        "note": "Uses v98-like train LOO full AD cache as the strongest available leakage-free full-H training backbone; not an online-score predictor.",
        "count": int(len(indices)),
        "indices": indices.tolist(),
        "device": str(device),
        "results": results,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    order = sorted(results, key=lambda k: float(results[k]["score_proxy"]), reverse=True)
    lines = [
        "# v124 Bottleneck Oracle Decomposition",
        "",
        "该表使用 feature-matched 训练 LOO 点，只用于比较剩余误差来源，不直接预测线上分数。",
        "",
        "| variant | score proxy | gain | PAS | PDP | global NMSE | joint HVT | full power | pol/UE block | coherence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in order:
        r = results[name]
        lines.append(
            f"| `{name}` | {r['score_proxy']:.6f} | {r['score_gain_vs_base']:+.6f} | "
            f"{r['pas']:.6f} | {r['pdp']:.6f} | {r['nmse_global']:.6f} | "
            f"{r['joint_hvt']['mean']:.6f} | {r['full_power']['mean']:.6f} | "
            f"{r['block_polue']['mean']:.6f} | {r['complex_coherence']['mean']:.6f} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md, "order": order}, indent=2))


if __name__ == "__main__":
    main()
