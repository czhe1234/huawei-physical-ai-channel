#!/usr/bin/env python3
"""Analyze candidate metric objects on full-H LOO bases.

This script compares two available train LOO bases against the same train truth:

* v36 derotated LOO AD base
* v98-like LOO AD base after support projection

The goal is to infer which candidate PAS/PDP/support objects are improved by
the support-only machinery that later led to v102-v115 online gains.

It uses the leaked platform PAS reshape:

    H -> reshape (B,2,16,8,4,192) -> transpose (B,2,192,16,4,8)
    np.fft.fft2(..., axes=(2,3))

and several conventional AD-support views.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import inverse_ad


ROOT = Path(__file__).resolve().parents[1]
ASSETS = PATHS.assets
TINY = 1e-30

MP = 2
MH = 16
MV = 8
N_ANT = 4
S_SC = 192


def summarize(v: list[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(v, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
    }


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(a.shape[0], -1).astype(np.float64)
    bb = b.reshape(b.shape[0], -1).astype(np.float64)
    num = np.sum(aa * bb, axis=1)
    den = np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1))
    return num / np.maximum(den, TINY)


def cosine_vectors(pred_p: np.ndarray, true_p: np.ndarray, vector_axes: tuple[int, ...]) -> np.ndarray:
    axes = tuple(int(a) for a in vector_axes)
    keep_axes = [a for a in range(pred_p.ndim) if a not in axes]
    order = keep_axes + list(axes)
    p = np.transpose(pred_p, order)
    t = np.transpose(true_p, order)
    batch = pred_p.shape[0]
    vlen = int(np.prod([pred_p.shape[a] for a in axes]))
    p = p.reshape(batch, -1, vlen)
    t = t.reshape(batch, -1, vlen)
    scores = row_cos(p.reshape(-1, vlen), t.reshape(-1, vlen))
    return scores.reshape(batch, -1).mean(axis=1)


def leaked_pas_power_from_ad(ad: np.ndarray) -> np.ndarray:
    h = inverse_ad(ad)
    arr = h.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    arr = arr.transpose(0, 1, 5, 2, 4, 3)  # B,2,192,16,4,8
    spec = np.fft.fft2(arr, axes=(2, 3))
    return (spec.real**2 + spec.imag**2).astype(np.float64)


def add_conventional_metrics(out: dict[str, list[float]], pred_ad: np.ndarray, true_ad: np.ndarray) -> None:
    pp = (pred_ad.real**2 + pred_ad.imag**2).astype(np.float64)  # B,2,H,V,4,T
    tp = (true_ad.real**2 + true_ad.imag**2).astype(np.float64)
    pc = pp.sum(axis=(1, 4))  # B,H,V,T
    tc = tp.sum(axis=(1, 4))

    out["conv_joint_hvt"].extend(row_cos(pc, tc).tolist())
    out["conv_beam_hv"].extend(row_cos(pc.sum(axis=3), tc.sum(axis=3)).tolist())
    out["conv_delay_t"].extend(row_cos(pc.sum(axis=(1, 2)), tc.sum(axis=(1, 2))).tolist())
    out["conv_h_delay_ht"].extend(row_cos(pc.sum(axis=2), tc.sum(axis=2)).tolist())
    out["conv_v_delay_vt"].extend(row_cos(pc.sum(axis=1), tc.sum(axis=1)).tolist())
    out["conv_h_only"].extend(row_cos(pc.sum(axis=(2, 3)), tc.sum(axis=(2, 3))).tolist())
    out["conv_v_only"].extend(row_cos(pc.sum(axis=(1, 3)), tc.sum(axis=(1, 3))).tolist())
    out["conv_pol_frac"].extend(row_cos(pp.sum(axis=(2, 3, 4, 5)), tp.sum(axis=(2, 3, 4, 5))).tolist())
    out["conv_ue_frac"].extend(row_cos(pp.sum(axis=(1, 2, 3, 5)), tp.sum(axis=(1, 2, 3, 5))).tolist())

    # Per-block objects.  These expose whether collapsing pol/UE/V hides errors.
    # pp dims: B,pol,H,V,UE,T
    out["conv_hvt_per_polue"].extend(cosine_vectors(pp.transpose(0, 1, 4, 2, 3, 5), tp.transpose(0, 1, 4, 2, 3, 5), (3, 4, 5)).tolist())
    out["conv_ht_per_polue_v"].extend(cosine_vectors(pp.transpose(0, 1, 4, 3, 2, 5), tp.transpose(0, 1, 4, 3, 2, 5), (4, 5)).tolist())
    out["conv_vt_per_polue_h"].extend(cosine_vectors(pp.transpose(0, 1, 4, 2, 3, 5), tp.transpose(0, 1, 4, 2, 3, 5), (4, 5)).tolist())


def add_leaked_metrics(out: dict[str, list[float]], pred_ad: np.ndarray, true_ad: np.ndarray) -> None:
    pp = leaked_pas_power_from_ad(pred_ad)  # B,pol,S,H,UE,V
    tp = leaked_pas_power_from_ad(true_ad)

    out["leaked_all"].extend(cosine_vectors(pp, tp, (1, 2, 3, 4, 5)).tolist())
    out["leaked_sh_per_pol_ue_v"].extend(cosine_vectors(pp, tp, (2, 3)).tolist())
    out["leaked_shv_per_pol_ue"].extend(cosine_vectors(pp, tp, (2, 3, 5)).tolist())
    out["leaked_sv_per_pol_ue_h"].extend(cosine_vectors(pp, tp, (2, 5)).tolist())
    out["leaked_hv_per_pol_ue_s"].extend(cosine_vectors(pp, tp, (3, 5)).tolist())
    out["leaked_s_per_pol_ue_hv"].extend(cosine_vectors(pp, tp, (2,)).tolist())
    out["leaked_h_per_pol_ue_sv"].extend(cosine_vectors(pp, tp, (3,)).tolist())
    out["leaked_sum_pol_shv_per_ue"].extend(cosine_vectors(pp.sum(axis=1), tp.sum(axis=1), (1, 2, 4)).tolist())
    out["leaked_sum_ue_shv_per_pol"].extend(cosine_vectors(pp.sum(axis=4), tp.sum(axis=4), (2, 3, 4)).tolist())
    out["leaked_sum_polue_shv"].extend(cosine_vectors(pp.sum(axis=(1, 4)), tp.sum(axis=(1, 4)), (1, 2, 3)).tolist())


def analyze_pair(name: str, base_path: Path, residual_path: Path, batch: int) -> dict[str, object]:
    base = np.load(base_path, mmap_mode="r")
    residual = np.load(residual_path, mmap_mode="r")
    if base.shape != residual.shape:
        raise ValueError(f"shape mismatch for {name}: {base.shape} vs {residual.shape}")
    metrics: dict[str, list[float]] = {}
    for key in [
        "conv_joint_hvt",
        "conv_beam_hv",
        "conv_delay_t",
        "conv_h_delay_ht",
        "conv_v_delay_vt",
        "conv_h_only",
        "conv_v_only",
        "conv_pol_frac",
        "conv_ue_frac",
        "conv_hvt_per_polue",
        "conv_ht_per_polue_v",
        "conv_vt_per_polue_h",
        "leaked_all",
        "leaked_sh_per_pol_ue_v",
        "leaked_shv_per_pol_ue",
        "leaked_sv_per_pol_ue_h",
        "leaked_hv_per_pol_ue_s",
        "leaked_s_per_pol_ue_hv",
        "leaked_h_per_pol_ue_sv",
        "leaked_sum_pol_shv_per_ue",
        "leaked_sum_ue_shv_per_pol",
        "leaked_sum_polue_shv",
    ]:
        metrics[key] = []

    n = base.shape[0]
    for start in range(0, n, batch):
        end = min(start + batch, n)
        pred = np.asarray(base[start:end], dtype=np.complex64)
        truth = (pred + np.asarray(residual[start:end], dtype=np.complex64)).astype(np.complex64)
        add_conventional_metrics(metrics, pred, truth)
        add_leaked_metrics(metrics, pred, truth)
        print(f"{name}: {end}/{n}", flush=True)

    return {
        "name": name,
        "base_path": str(base_path),
        "residual_path": str(residual_path),
        "shape": list(base.shape),
        "metrics": {k: summarize(v) for k, v in metrics.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--out-json", default=str(ASSETS / "metric_hypothesis_fullh_inference.json"))
    ap.add_argument("--out-md", default=str(ASSETS / "metric_hypothesis_fullh_inference.md"))
    args = ap.parse_args()

    pairs = [
        (
            "v36_loo_base",
            ASSETS / "train_v36_loo_base_derot_ad.npy",
            ASSETS / "train_v36_loo_residual_derot_ad.npy",
        ),
        (
            "v98like_loo_base",
            ASSETS / "train_v98like_loo_base_ad.npy",
            ASSETS / "train_v98like_loo_residual_ad.npy",
        ),
    ]
    results = []
    for name, base, residual in pairs:
        if not base.exists() or not residual.exists():
            print(f"skip missing {name}: {base} / {residual}", flush=True)
            continue
        results.append(analyze_pair(name, base, residual, int(args.batch)))

    by_name = {r["name"]: r for r in results}
    delta_rows = []
    if "v36_loo_base" in by_name and "v98like_loo_base" in by_name:
        m0 = by_name["v36_loo_base"]["metrics"]
        m1 = by_name["v98like_loo_base"]["metrics"]
        for key in sorted(m0):
            delta_rows.append(
                {
                    "metric": key,
                    "v36_mean": m0[key]["mean"],
                    "v98like_mean": m1[key]["mean"],
                    "delta_mean": m1[key]["mean"] - m0[key]["mean"],
                    "v36_p10": m0[key]["p10"],
                    "v98like_p10": m1[key]["p10"],
                    "delta_p10": m1[key]["p10"] - m0[key]["p10"],
                }
            )
        delta_rows.sort(key=lambda x: x["delta_mean"], reverse=True)

    out = {
        "method": "analyze_metric_hypotheses_fullh",
        "results": results,
        "delta_rows": delta_rows,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Full-H Metric Hypothesis Inference",
        "",
        "比较 `v36_loo_base` 与 `v98like_loo_base` 相对训练真值的候选口径余弦。",
        "",
        "## Delta Sorted",
        "",
        "| metric | v36 mean | v98like mean | delta | v36 p10 | v98like p10 | p10 delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in delta_rows:
        lines.append(
            f"| `{row['metric']}` | {row['v36_mean']:.6f} | {row['v98like_mean']:.6f} | "
            f"{row['delta_mean']:+.6f} | {row['v36_p10']:.6f} | {row['v98like_p10']:.6f} | {row['delta_p10']:+.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- 如果 leaked 口径提升明显，说明平台报错暴露的 PAS 对 v102 后涨分有解释力。",
        "- 如果 conventional joint/conditional support 提升更明显，说明 v102 后主要仍是物理 AD support，而非 leaked PAS 本身。",
        "- p10 delta 比 mean delta 更重要：线上很多提升可能来自修复长尾差点，而不是平均样本。",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
