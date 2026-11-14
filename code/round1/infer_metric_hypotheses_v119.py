#!/usr/bin/env python3
"""Lightweight metric-hypothesis inference from train support targets.

This script does not try to fully reproduce the platform metric.  Its purpose is
to enumerate candidate "stable objects" and measure which ones are already
predictable by the strongest available train LOO support target.

The main input pair is:

* truth_coll_power.npy: true collapsed AD support, shape (2000, 16, 8, 192)
* train_v98like_loo_joint_target.npy: LOO predicted support target, same shape

If a candidate marginal/joint object is poorly predictable here, it is unlikely
to be the next high-upside target.  If it is predictable but not yet fully used
online, it becomes a good candidate for a new version.
"""

from __future__ import annotations
from project_paths import PATHS

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = PATHS.assets
TINY = 1e-30


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(a.shape[0], -1).astype(np.float64)
    bb = b.reshape(b.shape[0], -1).astype(np.float64)
    num = np.sum(aa * bb, axis=1)
    den = np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1))
    return num / np.maximum(den, TINY)


def normalize(x: np.ndarray, axes: tuple[int, ...] | None = None) -> np.ndarray:
    if axes is None:
        axes = tuple(range(1, x.ndim))
    return x / np.maximum(np.sum(x, axis=axes, keepdims=True), TINY)


def summarize(v: np.ndarray) -> dict[str, float]:
    x = np.asarray(v, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
    }


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    af = a.reshape(a.shape[0], -1)
    bf = b.reshape(b.shape[0], -1)
    ka = min(k, af.shape[1])
    ia = np.argpartition(af, -ka, axis=1)[:, -ka:]
    ib = np.argpartition(bf, -ka, axis=1)[:, -ka:]
    out = np.empty(af.shape[0], dtype=np.float32)
    for i in range(af.shape[0]):
        out[i] = len(set(ia[i].tolist()) & set(ib[i].tolist())) / float(ka)
    return out


def candidate_views(x: np.ndarray) -> dict[str, np.ndarray]:
    """Return normalized candidate metric objects from collapsed H,V,T support."""

    # x shape: B,H,V,T.  Normalize each object independently because PAS/PDP
    # cosines are shape comparisons, not total-power comparisons.
    views: dict[str, np.ndarray] = {}
    views["joint_hvt"] = normalize(x)
    views["beam_hv"] = normalize(x.sum(axis=3))
    views["delay_t"] = normalize(x.sum(axis=(1, 2)))
    views["h_delay_ht"] = normalize(x.sum(axis=2))
    views["v_delay_vt"] = normalize(x.sum(axis=1))
    views["h_only"] = normalize(x.sum(axis=(2, 3)))
    views["v_only"] = normalize(x.sum(axis=(1, 3)))
    views["h_given_delay"] = normalize(x.sum(axis=2), axes=(1,))  # B,H,T normalized over H per delay
    views["v_given_delay"] = normalize(x.sum(axis=1), axes=(1,))  # B,V,T normalized over V per delay
    views["delay_given_h"] = normalize(x.sum(axis=2), axes=(2,))  # B,H,T normalized over T per H
    views["delay_given_v"] = normalize(x.sum(axis=1), axes=(2,))  # B,V,T normalized over T per V
    return views


def main() -> None:
    truth_path = ASSETS / "truth_coll_power.npy"
    target_path = ASSETS / "train_v98like_loo_joint_target.npy"
    if not truth_path.exists() or not target_path.exists():
        raise FileNotFoundError("required support arrays are missing")

    truth = np.load(truth_path, mmap_mode="r").astype(np.float32)
    target = np.load(target_path, mmap_mode="r").astype(np.float32)
    if truth.shape != target.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {target.shape}")

    truth_views = candidate_views(np.asarray(truth, dtype=np.float32))
    target_views = candidate_views(np.asarray(target, dtype=np.float32))

    rows = []
    for name in truth_views:
        cos = row_cos(target_views[name], truth_views[name])
        rows.append({"name": name, "cos": summarize(cos)})

    # Top-k support identity is useful because recent versions improve by
    # moving energy among top AD components, not only by matching smooth
    # marginals.
    for k in (4, 16, 64, 256, 1024):
        ov = topk_overlap(target_views["joint_hvt"], truth_views["joint_hvt"], k)
        rows.append({"name": f"joint_top{k}_overlap", "overlap": summarize(ov)})

    rows_sorted = sorted(rows, key=lambda r: r.get("cos", r.get("overlap", {})).get("mean", -1), reverse=True)

    out = {
        "method": "infer_metric_hypotheses_v119",
        "truth": str(truth_path),
        "target": str(target_path),
        "shape": list(truth.shape),
        "rows": rows_sorted,
        "interpretation": [
            "High cosine means this candidate object is easy for current train LOO support target.",
            "High top-k overlap means component identity is predictable, not only smooth marginals.",
            "Objects with high predictability but not explicitly optimized online are good next targets.",
        ],
    }

    json_path = ASSETS / "metric_hypothesis_support_inference.json"
    md_path = ASSETS / "metric_hypothesis_support_inference.md"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Metric Hypothesis Support Inference",
        "",
        "输入：`truth_coll_power.npy` vs `train_v98like_loo_joint_target.npy`。",
        "",
        "| candidate object | mean | p10 | median | min |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows_sorted:
        stats = row.get("cos") or row.get("overlap")
        lines.append(
            f"| `{row['name']}` | {stats['mean']:.6f} | {stats['p10']:.6f} | "
            f"{stats['median']:.6f} | {stats['min']:.6f} |"
        )
    lines += [
        "",
        "解读：",
        "",
        "- `beam_hv` / `delay_t` 高，说明边缘 PAS/PDP 很容易预测，但它们过去单独优化没有带来大涨。",
        "- `joint_hvt` 和 top-k overlap 更关键，因为 v92-v115 的线上提升主要来自 joint support。",
        "- 如果某个条件分布如 `delay_given_h` 或 `h_given_delay` 明显高于完整 joint，可以作为下一版更细的投影对象。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
