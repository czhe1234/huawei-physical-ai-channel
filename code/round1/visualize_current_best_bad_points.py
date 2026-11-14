#!/usr/bin/env python3
"""Visualize current-best bad/risky point locations.

Training points have truth, so "bad" means the leave-one-out support
prediction is actually wrong.  Test points do not have truth; "risk" is only a
proxy from v121 template uncertainty and distance to the train support manifold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_map_features import BS_POS, read_ply_memmap
from generate_v112_kernel_moe_path_support import normalize_support, row_cos, standardize
from generate_v121_support_template_manifold import support_signature
from project_paths import PATHS
from powerfield_common import ASSETS, regime_of


REGIMES = ("clear", "light", "medium", "heavy")
TINY = 1e-12


def entropy01(w: np.ndarray) -> np.ndarray:
    return (-(w * np.log(np.maximum(w, TINY))).sum(axis=1) / math.log(w.shape[1])).astype(np.float32)


def sample_map(vertices, max_points: int, seed: int) -> dict[str, np.ndarray]:
    n = len(vertices)
    rng = np.random.default_rng(seed)
    count = min(max_points, n)
    idx = rng.choice(n, size=count, replace=False)
    return {
        "x": np.asarray(vertices["x"][idx], dtype=np.float32),
        "y": np.asarray(vertices["y"][idx], dtype=np.float32),
        "z": np.asarray(vertices["z"][idx], dtype=np.float32),
        "nz": np.asarray(vertices["nz"][idx], dtype=np.float32),
    }


def load_map_sample(path: Path, cache: Path, max_points: int, seed: int) -> dict[str, np.ndarray]:
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    vertices = read_ply_memmap(path)
    sample = sample_map(vertices, max_points, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **sample)
    return sample


def robust01(x: np.ndarray, q0: float = 0.05, q1: float = 0.95) -> np.ndarray:
    lo = float(np.quantile(x, q0))
    hi = float(np.quantile(x, q1))
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)


def finite_axes(ax, xlim=None, ylim=None) -> None:
    ax.scatter(
        [BS_POS[0]],
        [BS_POS[1]],
        marker="*",
        s=180,
        c="#8c00ff",
        edgecolors="white",
        linewidths=0.6,
        zorder=10,
        label="BS",
    )
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal", adjustable="box")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.12, linewidth=0.5)


def draw_wall_terrain(ax, sample, alpha: float = 0.22, wall_alpha: float = 0.28) -> None:
    ax.scatter(sample["x"], sample["y"], s=0.55, c="#d8d8d8", alpha=alpha * 0.45, linewidths=0, zorder=0)
    wall = (sample["z"] > 2.0) & (np.abs(sample["nz"]) < 0.7)
    ax.scatter(sample["x"][wall], sample["y"][wall], s=0.9, c="#242424", alpha=wall_alpha, linewidths=0, zorder=1)


def compute_train_metrics(args) -> dict[str, np.ndarray]:
    truth = normalize_support(np.load(args.truth_support, mmap_mode="r"), args.floor)
    pred = normalize_support(np.load(args.train_pred_support, mmap_mode="r"), args.floor)
    joint = row_cos(pred, truth).astype(np.float32)
    beam = row_cos(pred.sum(axis=-1), truth.sum(axis=-1)).astype(np.float32)
    delay = row_cos(pred.sum(axis=(1, 2)), truth.sum(axis=(1, 2))).astype(np.float32)
    score = (0.50 * joint + 0.25 * beam + 0.25 * delay).astype(np.float32)
    return {"score": score, "joint": joint, "beam": beam, "delay": delay, "badness": (1.0 - score).astype(np.float32)}


def nearest_signature_distance(train_support: np.ndarray, test_support: np.ndarray, floor: float, chunk: int) -> np.ndarray:
    train_sig = support_signature(train_support, floor)
    test_sig = support_signature(test_support, floor)
    train_sig, test_sig = standardize(train_sig, test_sig)
    train_norm = np.sum(train_sig * train_sig, axis=1)[None, :]
    out = np.empty(len(test_sig), dtype=np.float32)
    for s in range(0, len(test_sig), chunk):
        e = min(s + chunk, len(test_sig))
        q = test_sig[s:e]
        d = np.sum(q * q, axis=1, keepdims=True) + train_norm - 2.0 * (q @ train_sig.T)
        out[s:e] = np.sqrt(np.maximum(d.min(axis=1), 0.0)).astype(np.float32)
    return out


def compute_test_risk(args) -> dict[str, np.ndarray]:
    truth = normalize_support(np.load(args.truth_support, mmap_mode="r"), args.floor)
    target = normalize_support(np.load(args.v121_target, mmap_mode="r"), args.floor)
    probs = np.load(args.v121_probs, mmap_mode="r").astype(np.float32)
    prob_top1 = probs.max(axis=1).astype(np.float32)
    prob_entropy = entropy01(probs)
    sig_dist = nearest_signature_distance(truth, target, args.floor, args.signature_chunk)
    risk = (
        0.48 * robust01(prob_entropy)
        + 0.32 * robust01(sig_dist)
        + 0.20 * robust01(1.0 - prob_top1)
    ).astype(np.float32)
    return {
        "risk": risk,
        "prob_top1": prob_top1,
        "prob_entropy": prob_entropy,
        "signature_distance": sig_dist,
    }


def annotate_ids(ax, pos: np.ndarray, ids: np.ndarray, color: str, n: int = 30) -> None:
    for i in ids[:n]:
        ax.text(pos[i, 0], pos[i, 1], str(int(i)), fontsize=7, color=color, zorder=20, clip_on=True)


def plot_train_global(fig_dir: Path, sample, train_pos, train_metrics, bad_ids) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    ax = axes[0]
    draw_wall_terrain(ax, sample)
    sc = ax.scatter(train_pos[:, 0], train_pos[:, 1], s=17, c=train_metrics["badness"], cmap="magma", alpha=0.88, linewidths=0, zorder=4)
    ax.scatter(train_pos[bad_ids, 0], train_pos[bad_ids, 1], s=76, facecolors="none", edgecolors="#00d5ff", linewidths=1.4, zorder=8)
    annotate_ids(ax, train_pos, bad_ids, "#005a70")
    finite_axes(ax)
    ax.set_title("Train LOO true badness: 1 - support score")
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="badness")

    ax = axes[1]
    draw_wall_terrain(ax, sample, alpha=0.10, wall_alpha=0.22)
    sc = ax.scatter(train_pos[:, 0], train_pos[:, 1], s=17, c=train_metrics["score"], cmap="viridis", alpha=0.88, linewidths=0, zorder=4)
    ax.scatter(train_pos[bad_ids, 0], train_pos[bad_ids, 1], s=76, facecolors="none", edgecolors="#ff2020", linewidths=1.4, zorder=8)
    annotate_ids(ax, train_pos, bad_ids, "#7a0000")
    finite_axes(ax)
    ax.set_title("Train LOO support score, lower is worse")
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="support score")

    fig.savefig(fig_dir / "train_loo_bad_points_current_global.png", dpi=190)
    plt.close(fig)


def plot_test_risk_global(fig_dir: Path, sample, train_pos, test_pos, test_metrics, risk_ids) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    ax = axes[0]
    draw_wall_terrain(ax, sample)
    ax.scatter(train_pos[:, 0], train_pos[:, 1], s=5, c="#1f77b4", alpha=0.12, linewidths=0, label="train", zorder=3)
    sc = ax.scatter(test_pos[:, 0], test_pos[:, 1], s=30, c=test_metrics["risk"], cmap="magma", alpha=0.92, marker="x", linewidths=1.0, zorder=5)
    ax.scatter(test_pos[risk_ids, 0], test_pos[risk_ids, 1], s=82, facecolors="none", edgecolors="#ff2020", linewidths=1.6, zorder=8)
    annotate_ids(ax, test_pos, risk_ids, "#7a0000")
    finite_axes(ax)
    ax.set_title("Test high-risk proxy on terrain (truth unavailable)")
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="risk proxy")

    ax = axes[1]
    draw_wall_terrain(ax, sample, alpha=0.10, wall_alpha=0.24)
    sc = ax.scatter(test_pos[:, 0], test_pos[:, 1], s=30, c=test_metrics["signature_distance"], cmap="plasma", alpha=0.92, marker="x", linewidths=1.0, zorder=5)
    ax.scatter(test_pos[risk_ids, 0], test_pos[risk_ids, 1], s=82, facecolors="none", edgecolors="#00d5ff", linewidths=1.4, zorder=8)
    annotate_ids(ax, test_pos, risk_ids, "#005a70")
    finite_axes(ax)
    ax.set_title("Test distance to train support-manifold signature")
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="nearest signature distance")

    fig.savefig(fig_dir / "test_v121_high_risk_proxy_global.png", dpi=190)
    plt.close(fig)


def plot_overlay(fig_dir: Path, sample, train_pos, test_pos, bad_ids, risk_ids) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)
    draw_wall_terrain(ax, sample, alpha=0.16, wall_alpha=0.27)
    ax.scatter(train_pos[:, 0], train_pos[:, 1], s=5, c="#1f77b4", alpha=0.11, linewidths=0, label="all train", zorder=3)
    ax.scatter(test_pos[:, 0], test_pos[:, 1], s=13, c="#ff7f0e", alpha=0.28, marker="x", linewidths=0.8, label="all test", zorder=4)
    ax.scatter(train_pos[bad_ids, 0], train_pos[bad_ids, 1], s=72, facecolors="none", edgecolors="#00d5ff", linewidths=1.3, label="train true bad LOO", zorder=8)
    ax.scatter(test_pos[risk_ids, 0], test_pos[risk_ids, 1], s=82, facecolors="none", edgecolors="#ff2020", linewidths=1.5, label="test high-risk proxy", zorder=9)
    annotate_ids(ax, train_pos, bad_ids, "#005a70", n=20)
    annotate_ids(ax, test_pos, risk_ids, "#7a0000", n=24)
    finite_axes(ax)
    ax.set_title("Train true-bad points and test high-risk proxy points")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.savefig(fig_dir / "train_bad_and_test_risk_overlay.png", dpi=190)
    plt.close(fig)


def plot_metric_panels(fig_dir: Path, sample, train_pos, test_pos, train_metrics, test_metrics) -> None:
    panels = [
        ("train", train_pos, train_metrics["joint"], "Train joint AD cosine", "viridis", False),
        ("train", train_pos, train_metrics["beam"], "Train beam/PAS cosine", "viridis", False),
        ("train", train_pos, train_metrics["delay"], "Train delay/PDP cosine", "viridis", False),
        ("test", test_pos, test_metrics["prob_entropy"], "Test v121 template entropy", "magma", True),
        ("test", test_pos, test_metrics["prob_top1"], "Test v121 top template prob", "viridis", False),
        ("test", test_pos, test_metrics["risk"], "Test risk proxy", "magma", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for ax, (_kind, pos, vals, title, cmap, high_bad) in zip(axes.ravel(), panels):
        draw_wall_terrain(ax, sample, alpha=0.08, wall_alpha=0.16)
        sc = ax.scatter(pos[:, 0], pos[:, 1], s=16 if len(pos) > 1000 else 30, c=vals, cmap=cmap, alpha=0.88, marker="o" if len(pos) > 1000 else "x", linewidths=0.8 if len(pos) <= 1000 else 0, zorder=4)
        ids = np.argsort(vals)[-40:][::-1] if high_bad else np.argsort(vals)[:40]
        ax.scatter(pos[ids, 0], pos[ids, 1], s=62, facecolors="none", edgecolors="#ff2020", linewidths=1.1, zorder=8)
        finite_axes(ax)
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    fig.savefig(fig_dir / "bad_point_metric_panels_current.png", dpi=180)
    plt.close(fig)


def write_csv(path: Path, pos: np.ndarray, ids: np.ndarray, metrics: dict[str, np.ndarray], metric_keys: list[str], regimes: np.ndarray | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["rank", "index", "x", "y", "z"]
        if regimes is not None:
            header.append("regime")
        header.extend(metric_keys)
        w.writerow(header)
        for rank, i in enumerate(ids, 1):
            row = [rank, int(i), float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])]
            if regimes is not None:
                row.append(str(regimes[i]))
            row.extend(float(metrics[k][i]) for k in metric_keys)
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PATHS.analysis / "visualizations/current_best_bad_points"))
    ap.add_argument("--train-pos", default=str(PATHS.round1 / "Round1_Train_Pos.npy"))
    ap.add_argument("--test-pos", default=str(PATHS.round1 / "Round1_Test_Pos.npy"))
    ap.add_argument("--map", default=str(PATHS.round1 / "Round1_Map.ply"))
    ap.add_argument("--map-sample", default=str(PATHS.analysis / "visualizations/v98_bad_points/data/map_sample_420000_20260708.npz"))
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--train-pred-support", default=ASSETS + "/train_v98like_loo_joint_target.npy")
    ap.add_argument("--v121-target", default=ASSETS + "/cache_v121/v121_support_template_manifold_target.npy")
    ap.add_argument("--v121-probs", default=ASSETS + "/cache_v121/v121_support_template_manifold_probs.npy")
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--top-n", type=int, default=80)
    ap.add_argument("--risk-n", type=int, default=60)
    ap.add_argument("--signature-chunk", type=int, default=80)
    ap.add_argument("--max-map-points", type=int, default=420000)
    ap.add_argument("--seed", type=int, default=20260708)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_pos = np.load(args.train_pos)
    test_pos = np.load(args.test_pos)
    geo = np.load(args.geometry_cache)
    train_regimes = regime_of(geo["train_blocked"].astype(np.float64))
    sample = load_map_sample(Path(args.map), Path(args.map_sample), args.max_map_points, args.seed)

    train_metrics = compute_train_metrics(args)
    test_metrics = compute_test_risk(args)
    bad_ids = np.argsort(train_metrics["score"])[: args.top_n]
    risk_ids = np.argsort(test_metrics["risk"])[-args.risk_n :][::-1]

    plot_train_global(fig_dir, sample, train_pos, train_metrics, bad_ids)
    plot_test_risk_global(fig_dir, sample, train_pos, test_pos, test_metrics, risk_ids)
    plot_overlay(fig_dir, sample, train_pos, test_pos, bad_ids[:40], risk_ids[:40])
    plot_metric_panels(fig_dir, sample, train_pos, test_pos, train_metrics, test_metrics)

    write_csv(
        data_dir / "train_true_bad_top.csv",
        train_pos,
        bad_ids,
        train_metrics,
        ["score", "badness", "joint", "beam", "delay"],
        train_regimes,
    )
    write_csv(
        data_dir / "test_high_risk_proxy_top.csv",
        test_pos,
        risk_ids,
        test_metrics,
        ["risk", "prob_top1", "prob_entropy", "signature_distance"],
        None,
    )

    summary = {
        "method": "current_best_bad_point_visualization",
        "train_bad_definition": "truth-available LOO support score: 0.50*joint_AD_cos + 0.25*beam_cos + 0.25*delay_cos",
        "test_risk_definition": "no truth; proxy = entropy + distance from train support-signature manifold + low top-template probability",
        "train_worst_indices": bad_ids[:40].astype(int).tolist(),
        "test_highest_risk_indices": risk_ids[:40].astype(int).tolist(),
        "train_score": {
            "mean": float(train_metrics["score"].mean()),
            "p10": float(np.quantile(train_metrics["score"], 0.10)),
            "median": float(np.quantile(train_metrics["score"], 0.50)),
            "min": float(train_metrics["score"].min()),
        },
        "test_risk": {
            "mean": float(test_metrics["risk"].mean()),
            "p90": float(np.quantile(test_metrics["risk"], 0.90)),
            "max": float(test_metrics["risk"].max()),
        },
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = f"""# Current Best Bad / Risky Point Distribution

This folder contains static PNG figures for discussing where the current support-model family fails.

Important distinction:

- Train points are **true bad points** because leave-one-out support prediction can be compared with training truth.
- Test points are **high-risk proxy points** because test truth is not available. They indicate uncertainty or out-of-manifold support shape, not confirmed online errors.

## Figures

- `figures/train_loo_bad_points_current_global.png`: true train LOO bad points over official point-cloud terrain.
- `figures/test_v121_high_risk_proxy_global.png`: v121 test high-risk proxy points over terrain.
- `figures/train_bad_and_test_risk_overlay.png`: train true-bad points and test high-risk proxy points in one map.
- `figures/bad_point_metric_panels_current.png`: metric panels showing which component is bad.

## CSV

- `data/train_true_bad_top.csv`: top {args.top_n} true train bad points.
- `data/test_high_risk_proxy_top.csv`: top {args.risk_n} test high-risk proxy points.

## Key Lists

Train true-bad top 20:

```text
{bad_ids[:20].astype(int).tolist()}
```

Test high-risk proxy top 20:

```text
{risk_ids[:20].astype(int).tolist()}
```

## Definitions

```text
train_support_score = 0.50 * joint_AD_cos + 0.25 * beam_cos + 0.25 * delay_cos
train_badness = 1 - train_support_score
```

```text
test_risk_proxy = 0.48 * template_entropy_rank
                + 0.32 * train_support_manifold_distance_rank
                + 0.20 * low_top_template_probability_rank
```
"""
    (out_dir / "analysis.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
