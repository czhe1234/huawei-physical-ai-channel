#!/usr/bin/env python3
"""Dedicated terrain visualization for train LOO bad points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_map_features import BS_POS
from project_paths import PATHS
from powerfield_common import regime_of


def finite_axes(ax, xlim=None, ylim=None):
    ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=180, c="#8c00ff", edgecolors="white", linewidths=0.6, zorder=9)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal", adjustable="box")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.12, linewidth=0.5)


def draw_walls(ax, sample, alpha=0.24, s=0.9):
    wall = (sample["z"] > 2.0) & (np.abs(sample["nz"]) < 0.7)
    ax.scatter(sample["x"][wall], sample["y"][wall], s=s, c="#303030", alpha=alpha, linewidths=0, zorder=1)


def annotate_top(ax, pos, ids, color="#7a0000", n=25):
    for i in ids[:n]:
        ax.text(pos[i, 0], pos[i, 1], str(i), fontsize=7, color=color, zorder=20, clip_on=True)


def plot_global(fig_dir, sample, pos, metrics, bad_ids):
    score = metrics["score"]
    err = 1.0 - score
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    ax = axes[0]
    draw_walls(ax, sample, alpha=0.20, s=0.9)
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=18, c=err, cmap="magma", alpha=0.86, linewidths=0, zorder=4)
    ax.scatter(pos[bad_ids, 0], pos[bad_ids, 1], s=78, facecolors="none", edgecolors="#00d5ff", linewidths=1.4, zorder=8)
    annotate_top(ax, pos, bad_ids, color="#004c5c", n=30)
    finite_axes(ax)
    ax.set_title("Train LOO support error on terrain: 1 - score")
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="1 - support score")

    ax = axes[1]
    draw_walls(ax, sample, alpha=0.18, s=0.8)
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=18, c=score, cmap="viridis", alpha=0.86, linewidths=0, zorder=4)
    ax.scatter(pos[bad_ids, 0], pos[bad_ids, 1], s=78, facecolors="none", edgecolors="#ff2020", linewidths=1.4, zorder=8)
    annotate_top(ax, pos, bad_ids, color="#7a0000", n=30)
    finite_axes(ax)
    ax.set_title("Train LOO support score on terrain")
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="support score, higher is better")

    fig.savefig(fig_dir / "train_bad_points_terrain_global.png", dpi=190)
    plt.close(fig)


def plot_metric_panels(fig_dir, sample, pos, metrics, bad_ids):
    panels = [
        ("score", "Support score", "viridis", False),
        ("joint", "Joint AD cosine", "viridis", False),
        ("beam", "Beam/PAS cosine", "viridis", False),
        ("delay", "Delay/PDP cosine", "viridis", False),
        ("attention_entropy", "Attention entropy", "magma", True),
        ("attention_max", "Attention max weight", "viridis", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for ax, (key, title, cmap, high_bad) in zip(axes.ravel(), panels):
        draw_walls(ax, sample, alpha=0.12, s=0.6)
        vals = metrics[key]
        sc = ax.scatter(pos[:, 0], pos[:, 1], s=13, c=vals, cmap=cmap, alpha=0.84, linewidths=0, zorder=4)
        if high_bad:
            ids = np.argsort(vals)[-40:][::-1]
        else:
            ids = np.argsort(vals)[:40]
        ax.scatter(pos[ids, 0], pos[ids, 1], s=58, facecolors="none", edgecolors="#ff2020", linewidths=1.1, zorder=8)
        finite_axes(ax)
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    fig.savefig(fig_dir / "train_bad_points_metric_panels.png", dpi=180)
    plt.close(fig)


def plot_closeups(fig_dir, sample, pos, metrics, bad_ids):
    boxes = [
        ("southwest low-score cluster", (42, 105), (-205, -145)),
        ("south/central low-score band", (82, 172), (-125, -45)),
        ("east edge low-score cluster", (205, 258), (-60, 45)),
        ("north/east low-score cluster", (145, 255), (20, 105)),
    ]
    err = 1.0 - metrics["score"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), constrained_layout=True)
    for ax, (title, xlim, ylim) in zip(axes.ravel(), boxes):
        local = (sample["x"] >= xlim[0]) & (sample["x"] <= xlim[1]) & (sample["y"] >= ylim[0]) & (sample["y"] <= ylim[1])
        wall = local & (sample["z"] > 2.0) & (np.abs(sample["nz"]) < 0.7)
        ax.scatter(sample["x"][local], sample["y"][local], s=0.6, c="#d8d8d8", alpha=0.14, linewidths=0, zorder=0)
        ax.scatter(sample["x"][wall], sample["y"][wall], s=1.2, c="#1f1f1f", alpha=0.32, linewidths=0, zorder=1)
        in_box = (pos[:, 0] >= xlim[0]) & (pos[:, 0] <= xlim[1]) & (pos[:, 1] >= ylim[0]) & (pos[:, 1] <= ylim[1])
        sc = ax.scatter(pos[in_box, 0], pos[in_box, 1], s=18, c=err[in_box], cmap="magma", alpha=0.82, linewidths=0, zorder=3)
        local_bad = [i for i in bad_ids if xlim[0] <= pos[i, 0] <= xlim[1] and ylim[0] <= pos[i, 1] <= ylim[1]]
        if local_bad:
            ids = np.asarray(local_bad, dtype=np.int64)
            ax.scatter(pos[ids, 0], pos[ids, 1], s=80, facecolors="none", edgecolors="#00d5ff", linewidths=1.4, zorder=8)
            annotate_top(ax, pos, ids, color="#005a70", n=30)
        finite_axes(ax, xlim=xlim, ylim=ylim)
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="1 - score")
    fig.savefig(fig_dir / "train_bad_points_terrain_closeups.png", dpi=190)
    plt.close(fig)


def write_top_csv(out_dir, pos, metrics, regimes, bad_ids, top_n):
    out = out_dir / "data" / "train_bad_points_top.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "index", "x", "y", "z", "regime", "score", "joint", "beam", "delay", "attention_max", "attention_entropy"])
        for rank, i in enumerate(bad_ids[:top_n], 1):
            w.writerow([
                rank,
                int(i),
                float(pos[i, 0]),
                float(pos[i, 1]),
                float(pos[i, 2]),
                str(regimes[i]),
                float(metrics["score"][i]),
                float(metrics["joint"][i]),
                float(metrics["beam"][i]),
                float(metrics["delay"][i]),
                float(metrics["attention_max"][i]),
                float(metrics["attention_entropy"][i]),
            ])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PATHS.analysis / "visualizations/v98_bad_points"))
    ap.add_argument("--train-pos", default=str(PATHS.round1 / "Round1_Train_Pos.npy"))
    ap.add_argument("--metrics", default=str(PATHS.analysis / "visualizations/v98_bad_points/data/train_loo_metrics.npz"))
    ap.add_argument("--map-sample", default=str(PATHS.analysis / "visualizations/v98_bad_points/data/map_sample_420000_20260708.npz"))
    ap.add_argument("--geometry-cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    ap.add_argument("--top-n", type=int, default=80)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    pos = np.load(args.train_pos)
    metrics_npz = np.load(args.metrics)
    metrics = {k: metrics_npz[k] for k in metrics_npz.files}
    sample_npz = np.load(args.map_sample)
    sample = {k: sample_npz[k] for k in sample_npz.files}
    geo = np.load(args.geometry_cache)
    regimes = regime_of(geo["train_blocked"].astype(np.float64))
    bad_ids = np.argsort(metrics["score"])[: int(args.top_n)]

    plot_global(fig_dir, sample, pos, metrics, bad_ids)
    plot_metric_panels(fig_dir, sample, pos, metrics, bad_ids)
    plot_closeups(fig_dir, sample, pos, metrics, bad_ids)
    csv_path = write_top_csv(out_dir, pos, metrics, regimes, bad_ids, int(args.top_n))

    md = f"""# Train LOO Bad Points

These figures show only training points, where true support error is available through leave-one-out prediction.

## Figures

- `figures/train_bad_points_terrain_global.png`: global terrain + training support error.
- `figures/train_bad_points_metric_panels.png`: score/joint/beam/delay/attention diagnostics.
- `figures/train_bad_points_terrain_closeups.png`: local zoom-ins around major bad clusters.

## CSV

- `{csv_path.relative_to(out_dir)}`: top {args.top_n} worst train points with coordinates and metrics.

## Worst train indices

```text
{bad_ids[:40].tolist()}
```

Definition:

```text
support_score = 0.50 * joint_AD_cos + 0.25 * beam_cos + 0.25 * delay_cos
badness = 1 - support_score
```
"""
    (out_dir / "train_bad_points_analysis.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
