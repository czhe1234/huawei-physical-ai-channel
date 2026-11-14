#!/usr/bin/env python3
"""Overlay v98 bad/risky points on the Round1 point-cloud terrain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_map_features import BS_POS, read_ply_memmap
from project_paths import PATHS


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


def overlay_points(ax, train_pos, test_pos, train_bad, test_bad, label_bad=True, draw_lines=False):
    ax.scatter(train_pos[:, 0], train_pos[:, 1], s=5, c="#1f77b4", alpha=0.16, linewidths=0, label="all train", zorder=3)
    ax.scatter(test_pos[:, 0], test_pos[:, 1], s=10, c="#ff7f0e", alpha=0.24, marker="x", linewidths=0.7, label="all test", zorder=4)
    ax.scatter(
        train_pos[train_bad, 0],
        train_pos[train_bad, 1],
        s=62,
        facecolors="none",
        edgecolors="#00d5ff",
        linewidths=1.3,
        label="train true bad LOO",
        zorder=7,
    )
    ax.scatter(
        test_pos[test_bad, 0],
        test_pos[test_bad, 1],
        s=74,
        facecolors="none",
        edgecolors="#ff2020",
        linewidths=1.5,
        label="test high-risk proxy",
        zorder=8,
    )
    if draw_lines:
        for i in test_bad[:20]:
            ax.plot([BS_POS[0], test_pos[i, 0]], [BS_POS[1], test_pos[i, 1]], color="#ff2020", alpha=0.18, linewidth=0.7, zorder=2)
    if label_bad:
        for i in train_bad[:16]:
            ax.text(train_pos[i, 0], train_pos[i, 1], str(i), fontsize=7, color="#005a70", zorder=10, clip_on=True)
        for i in test_bad[:20]:
            ax.text(test_pos[i, 0], test_pos[i, 1], str(i), fontsize=7, color="#7a0000", zorder=10, clip_on=True)


def plot_global(fig_dir: Path, sample, train_pos, test_pos, train_bad, test_bad):
    wall_mask = (sample["z"] > 2.0) & (np.abs(sample["nz"]) < 0.7)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    ax = axes[0]
    sc = ax.scatter(sample["x"], sample["y"], s=0.8, c=sample["z"], cmap="terrain", alpha=0.20, linewidths=0, zorder=1)
    overlay_points(ax, train_pos, test_pos, train_bad, test_bad, label_bad=True, draw_lines=False)
    finite_axes(ax)
    ax.set_title("Point-cloud terrain height + v98 bad/risky points")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="point-cloud z / m")

    ax = axes[1]
    ax.scatter(sample["x"][wall_mask], sample["y"][wall_mask], s=0.9, c="#2f2f2f", alpha=0.20, linewidths=0, zorder=1)
    overlay_points(ax, train_pos, test_pos, train_bad, test_bad, label_bad=True, draw_lines=True)
    finite_axes(ax)
    ax.set_title("Vertical-surface footprint + BS rays to test high-risk points")
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    out = fig_dir / "terrain_bad_points_global.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)


def plot_closeups(fig_dir: Path, sample, train_pos, test_pos, train_bad, test_bad):
    wall_mask = (sample["z"] > 2.0) & (np.abs(sample["nz"]) < 0.7)
    boxes = [
        ("southwest cluster", (42, 105), (-205, -145)),
        ("south/central cluster", (85, 170), (-125, -45)),
        ("east edge cluster", (205, 258), (-60, 45)),
        ("north/east cluster", (145, 255), (20, 105)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), constrained_layout=True)
    for ax, (title, xlim, ylim) in zip(axes.ravel(), boxes):
        local = wall_mask & (sample["x"] >= xlim[0]) & (sample["x"] <= xlim[1]) & (sample["y"] >= ylim[0]) & (sample["y"] <= ylim[1])
        ground = (sample["x"] >= xlim[0]) & (sample["x"] <= xlim[1]) & (sample["y"] >= ylim[0]) & (sample["y"] <= ylim[1])
        ax.scatter(sample["x"][ground], sample["y"][ground], s=0.6, c="#d8d8d8", alpha=0.16, linewidths=0, zorder=0)
        ax.scatter(sample["x"][local], sample["y"][local], s=1.2, c="#1c1c1c", alpha=0.30, linewidths=0, zorder=1)
        overlay_points(ax, train_pos, test_pos, train_bad, test_bad, label_bad=True, draw_lines=False)
        finite_axes(ax, xlim=xlim, ylim=ylim)
        ax.set_title(title)
    out = fig_dir / "terrain_bad_points_closeups.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)


def plot_density(fig_dir: Path, sample, train_pos, test_pos, train_bad, test_bad):
    wall_mask = (sample["z"] > 2.0) & (np.abs(sample["nz"]) < 0.7)
    fig, ax = plt.subplots(1, 1, figsize=(10, 9), constrained_layout=True)
    h = ax.hist2d(sample["x"][wall_mask], sample["y"][wall_mask], bins=180, cmap="Greys", cmin=1, alpha=0.92)
    overlay_points(ax, train_pos, test_pos, train_bad, test_bad, label_bad=True, draw_lines=False)
    finite_axes(ax)
    ax.set_title("Vertical-surface density heatmap + v98 bad/risky points")
    plt.colorbar(h[3], ax=ax, fraction=0.035, pad=0.02, label="wall point count / bin")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    out = fig_dir / "terrain_bad_points_wall_density.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(PATHS.round1 / "Round1_Map.ply"))
    ap.add_argument("--train-pos", default=str(PATHS.round1 / "Round1_Train_Pos.npy"))
    ap.add_argument("--test-pos", default=str(PATHS.round1 / "Round1_Test_Pos.npy"))
    ap.add_argument("--summary", default=str(PATHS.analysis / "visualizations/v98_bad_points/data/summary.json"))
    ap.add_argument("--out-dir", default=str(PATHS.analysis / "visualizations/v98_bad_points"))
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
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    train_bad = np.asarray(summary["train_worst_score_indices"][:40], dtype=np.int64)
    test_bad = np.asarray(summary["test_highest_risk_indices"][:40], dtype=np.int64)
    sample = load_map_sample(Path(args.map), data_dir / f"map_sample_{args.max_map_points}_{args.seed}.npz", args.max_map_points, args.seed)

    plot_global(fig_dir, sample, train_pos, test_pos, train_bad, test_bad)
    plot_closeups(fig_dir, sample, train_pos, test_pos, train_bad, test_bad)
    plot_density(fig_dir, sample, train_pos, test_pos, train_bad, test_bad)

    md = f"""# v98 Bad / Risky Points on Terrain

These figures overlay v98 train true-bad points and test high-risk proxy points on the official `Round1_Map.ply` point cloud.

## Figures

- `figures/terrain_bad_points_global.png`: terrain height and vertical-surface footprint, with train/test bad/risky points.
- `figures/terrain_bad_points_closeups.png`: four local zoom-ins around the main bad/risky clusters.
- `figures/terrain_bad_points_wall_density.png`: wall-density heatmap, useful for seeing building edges and narrow gaps.

## Legend

- Cyan circles: train LOO true-bad points, known from training truth.
- Red circles: test high-risk proxy points, not confirmed errors because test truth is unavailable.
- Orange x: all test receiver positions.
- Blue dots: all train receiver positions.
- Purple star: BS position.

## Current bad/risky index lists

Train true-bad top indices:

```text
{summary['train_worst_score_indices'][:20]}
```

Test high-risk proxy top indices:

```text
{summary['test_highest_risk_indices'][:20]}
```

"""
    (out_dir / "terrain_analysis.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
