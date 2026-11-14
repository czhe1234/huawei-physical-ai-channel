#!/usr/bin/env python3
"""Trace simple LOS + one-bounce paths for train bad points.

This is a diagnostic visualization, not a submission generator.  It uses only
official point-cloud-derived wall patches/candidates from
``official_pointcloud_scene_cache.npz`` and overlays simple geometric paths for
training points whose leave-one-out support prediction is poor.

The first diagnostic version only checked specular reflection geometry, so some
drawn paths could cross buildings.  This version additionally samples each
path segment against the official point-cloud height/wall grid and marks
blocked paths instead of treating them as valid rays.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_map_features import BS_POS, read_ply_memmap
from project_paths import PATHS
from powerfield_common import ASSETS, regime_of


TINY = 1e-12


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


def load_bad_ids(csv_path: Path, top_n: int) -> np.ndarray:
    ids: list[int] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["index"]))
            if len(ids) >= top_n:
                break
    return np.asarray(ids, dtype=np.int64)


def top_components(power: np.ndarray, k: int) -> list[dict[str, float]]:
    flat = power.reshape(-1)
    kk = min(k, flat.size)
    ids = np.argpartition(flat, -kk)[-kk:]
    ids = ids[np.argsort(flat[ids])[::-1]]
    total = float(flat.sum())
    out = []
    for rank, fid in enumerate(ids, 1):
        h = int(fid // (8 * 192))
        rem = int(fid % (8 * 192))
        v = int(rem // 192)
        t = int(rem % 192)
        out.append(
            {
                "rank": rank,
                "h": h,
                "v": v,
                "delay": t,
                "power": float(flat[fid]),
                "frac": float(flat[fid] / max(total, TINY)),
            }
        )
    return out


def finite_axes(ax, xlim, ylim) -> None:
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
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12, linewidth=0.5)


def draw_local_map(ax, sample, pos_xy, span: float) -> None:
    xlim = (pos_xy[0] - span, pos_xy[0] + span)
    ylim = (pos_xy[1] - span, pos_xy[1] + span)
    local = (
        (sample["x"] >= xlim[0])
        & (sample["x"] <= xlim[1])
        & (sample["y"] >= ylim[0])
        & (sample["y"] <= ylim[1])
    )
    wall = local & (sample["z"] > 2.0) & (np.abs(sample["nz"]) < 0.7)
    ax.scatter(sample["x"][local], sample["y"][local], s=0.7, c="#d8d8d8", alpha=0.12, linewidths=0, zorder=0)
    ax.scatter(sample["x"][wall], sample["y"][wall], s=1.3, c="#202020", alpha=0.34, linewidths=0, zorder=1)
    finite_axes(ax, xlim, ylim)


def local_limits_for_paths(rx: np.ndarray, refl_points: np.ndarray, span: float) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = [BS_POS[:2], rx[:2]]
    for rp in refl_points:
        if np.isfinite(rp[:2]).all():
            pts.append(rp[:2])
    arr = np.asarray(pts, dtype=np.float64)
    lo = arr.min(axis=0) - span
    hi = arr.max(axis=0) + span
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def nearest_delay_error(path_delay: float, comps: list[dict[str, float]]) -> float:
    if not comps or not np.isfinite(path_delay):
        return float("nan")
    delays = np.asarray([c["delay"] for c in comps], dtype=np.float64)
    return float(np.min(np.abs(delays - path_delay)))


def segment_blockage(
    a: np.ndarray,
    b: np.ndarray,
    scene: np.lib.npyio.NpzFile,
    samples: int,
    vertical_threshold: int,
    z_margin: float,
    t0: float,
    t1: float,
) -> tuple[bool, float, int]:
    """Lightweight point-cloud grid occlusion test for one segment."""

    t = np.linspace(float(t0), float(t1), int(samples), dtype=np.float64)
    pts = a[None, :] + t[:, None] * (b[None, :] - a[None, :])
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    res = float(scene["resolution"])
    nx = int(scene["grid_nx"])
    ny = int(scene["grid_ny"])
    ix = np.floor((pts[:, 0] - x_min) / res).astype(np.int64)
    iy = np.floor((pts[:, 1] - y_min) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.any(ok):
        return False, float("inf"), 0
    ix = ix[ok]
    iy = iy[ok]
    z_ray = pts[ok, 2]
    z_max = scene["grid_z_max"][iy, ix].astype(np.float64)
    vertical = scene["grid_vertical_count"][iy, ix].astype(np.int32)
    blocked = (vertical >= int(vertical_threshold)) & np.isfinite(z_max) & (z_max >= z_ray - float(z_margin))
    if not np.any(blocked):
        return False, float(np.nanmin(z_ray - z_max)), 0
    return True, float(np.nanmin(z_ray[blocked] - z_max[blocked])), int(np.count_nonzero(blocked))


def path_clearance(
    rx: np.ndarray,
    reflect_point: np.ndarray | None,
    scene: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
) -> dict[str, float | bool | int]:
    """Check LOS or two reflection segments and return blocked metadata."""

    if reflect_point is None:
        b, clearance, hits = segment_blockage(
            BS_POS.astype(np.float64),
            rx.astype(np.float64),
            scene,
            args.occlusion_samples,
            args.vertical_threshold,
            args.z_margin,
            0.02,
            0.98,
        )
        return {
            "blocked": bool(b),
            "blocked_bs_ref": bool(b),
            "blocked_ref_ue": False,
            "min_clearance": float(clearance),
            "block_hits": int(hits),
        }

    rp = reflect_point.astype(np.float64)
    b1, c1, h1 = segment_blockage(
        BS_POS.astype(np.float64),
        rp,
        scene,
        args.occlusion_samples,
        args.vertical_threshold,
        args.z_margin,
        0.02,
        0.90,
    )
    b2, c2, h2 = segment_blockage(
        rp,
        rx.astype(np.float64),
        scene,
        args.occlusion_samples,
        args.vertical_threshold,
        args.z_margin,
        0.10,
        0.98,
    )
    return {
        "blocked": bool(b1 or b2),
        "blocked_bs_ref": bool(b1),
        "blocked_ref_ue": bool(b2),
        "min_clearance": float(min(c1, c2)),
        "block_hits": int(h1 + h2),
    }


def plot_point(
    fig_dir: Path,
    idx: int,
    rx: np.ndarray,
    score_row: dict[str, float],
    sample: dict[str, np.ndarray],
    scene: np.lib.npyio.NpzFile,
    truth_power: np.ndarray,
    delay_slope: float,
    delay_intercept: float,
    reflect_count: int,
    span: float,
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, float]]]:
    refl_score = scene["train_reflection_score"][idx]
    refl_path = scene["train_reflection_path_length"][idx]
    refl_excess = scene["train_reflection_excess_length"][idx]
    refl_point = scene["train_reflection_point"][idx]
    refl_patch = scene["train_reflection_patch_id"][idx]

    comps = top_components(truth_power[idx], 12)
    path_rows: list[dict[str, float]] = []
    los_len = float(np.linalg.norm(rx - BS_POS))
    los_delay = float(delay_slope * los_len + delay_intercept)
    path_rows.append(
        {
            "point": int(idx),
            "path_rank": 0,
            "type": "LOS",
            "patch_id": -1,
            "score": 1.0,
            "length": los_len,
            "excess": 0.0,
            "delay_est": los_delay,
            "nearest_true_delay_error": nearest_delay_error(los_delay, comps),
            **path_clearance(rx, None, scene, args),
            "reflect_x": float("nan"),
            "reflect_y": float("nan"),
            "reflect_z": float("nan"),
        }
    )
    clear_reflections: list[int] = []
    for j in range(min(args.candidate_pool, refl_score.shape[0])):
        delay = float(delay_slope * float(refl_path[j]) + delay_intercept)
        clearance = path_clearance(rx, refl_point[j], scene, args)
        path_rows.append(
            {
                "point": int(idx),
                "path_rank": int(j + 1),
                "type": "one_bounce",
                "patch_id": int(refl_patch[j]),
                "score": float(refl_score[j]),
                "length": float(refl_path[j]),
                "excess": float(refl_excess[j]),
                "delay_est": delay,
                "nearest_true_delay_error": nearest_delay_error(delay, comps),
                **clearance,
                "reflect_x": float(refl_point[j, 0]),
                "reflect_y": float(refl_point[j, 1]),
                "reflect_z": float(refl_point[j, 2]),
            }
        )
        if not bool(clearance["blocked"]) and len(clear_reflections) < reflect_count:
            clear_reflections.append(j)

    refl_for_limits = refl_point[clear_reflections] if clear_reflections else refl_point[:1]
    xlim, ylim = local_limits_for_paths(rx, refl_for_limits, span)
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.0, 1.05])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_pdp = fig.add_subplot(gs[0, 1])
    ax_beam = fig.add_subplot(gs[0, 2])
    ax_table = fig.add_subplot(gs[1, 1:])

    local = (
        (sample["x"] >= xlim[0])
        & (sample["x"] <= xlim[1])
        & (sample["y"] >= ylim[0])
        & (sample["y"] <= ylim[1])
    )
    wall = local & (sample["z"] > 2.0) & (np.abs(sample["nz"]) < 0.7)
    ax_map.scatter(sample["x"][local], sample["y"][local], s=0.7, c="#d8d8d8", alpha=0.12, linewidths=0, zorder=0)
    ax_map.scatter(sample["x"][wall], sample["y"][wall], s=1.3, c="#202020", alpha=0.34, linewidths=0, zorder=1)
    finite_axes(ax_map, xlim, ylim)
    ax_map.scatter([rx[0]], [rx[1]], marker="o", s=90, c="#006eff", edgecolors="white", linewidths=0.6, label=f"train {idx}", zorder=10)
    los_blocked = bool(path_rows[0]["blocked"])
    ax_map.plot(
        [BS_POS[0], rx[0]],
        [BS_POS[1], rx[1]],
        color="#777777" if los_blocked else "#111111",
        lw=1.3,
        ls="--" if los_blocked else "-",
        alpha=0.58 if los_blocked else 0.72,
        label="LOS blocked" if los_blocked else "LOS clear",
        zorder=4,
    )
    colors = ["#e41a1c", "#ff7f00", "#984ea3", "#4daf4a", "#377eb8", "#a65628"]
    for draw_rank, j in enumerate(clear_reflections):
        rp = refl_point[j]
        if not np.isfinite(rp[:2]).all():
            continue
        col = colors[draw_rank % len(colors)]
        ax_map.plot([BS_POS[0], rp[0]], [BS_POS[1], rp[1]], color=col, lw=1.0, alpha=0.78, zorder=5)
        ax_map.plot([rp[0], rx[0]], [rp[1], rx[1]], color=col, lw=1.0, alpha=0.78, zorder=5, label=f"R{j+1} clear")
        ax_map.scatter([rp[0]], [rp[1]], marker="x", s=60, c=col, linewidths=1.4, zorder=8)
        ax_map.text(rp[0], rp[1], f"R{j+1}", fontsize=8, color=col, zorder=12, clip_on=True)
    ax_map.set_title(
        f"Train {idx}: occlusion-aware LOS + one-bounce paths\n"
        f"LOO support score={score_row['score']:.3f}, regime={score_row['regime']}"
    )
    ax_map.legend(loc="upper left", fontsize=8, frameon=True)

    pdp = truth_power[idx].sum(axis=(0, 1))
    pdp_db = 10.0 * np.log10(np.maximum(pdp / max(float(pdp.max()), TINY), 1e-12))
    ax_pdp.plot(pdp_db, color="#333333", lw=1.2)
    ax_pdp.axvline(
        los_delay,
        color="#777777" if los_blocked else "#111111",
        ls="--",
        lw=1.0,
        label=f"LOS {'blocked' if los_blocked else 'clear'} {los_delay:.1f}",
    )
    for draw_rank, j in enumerate(clear_reflections):
        d = float(delay_slope * float(refl_path[j]) + delay_intercept)
        ax_pdp.axvline(d, color=colors[draw_rank % len(colors)], ls=":", lw=1.1, label=f"R{j+1} {d:.1f}")
    ax_pdp.set_title("True AD PDP + geometric delay estimates")
    ax_pdp.set_xlabel("delay bin")
    ax_pdp.set_ylabel("relative dB")
    ax_pdp.set_ylim(-80, 2)
    ax_pdp.grid(alpha=0.15)
    ax_pdp.legend(fontsize=7, loc="upper right")

    beam = truth_power[idx].sum(axis=-1)
    beam_db = 10.0 * np.log10(np.maximum(beam / max(float(beam.max()), TINY), 1e-12))
    im = ax_beam.imshow(beam_db, origin="lower", aspect="auto", cmap="viridis", vmin=-50, vmax=0)
    ax_beam.set_title("True AD beam power")
    ax_beam.set_xlabel("v bin")
    ax_beam.set_ylabel("h bin")
    fig.colorbar(im, ax=ax_beam, fraction=0.046, pad=0.03, label="dB")

    ax_table.axis("off")
    table_rows = []
    for r in path_rows:
        table_rows.append(
            [
                r["path_rank"],
                r["type"],
                "yes" if not r["blocked"] else "no",
                r["patch_id"],
                f"{r['score']:.3f}",
                f"{r['length']:.1f}",
                f"{r['excess']:.1f}",
                f"{r['delay_est']:.1f}",
                f"{r['nearest_true_delay_error']:.1f}",
            ]
        )
    top_text = ", ".join([f"#{c['rank']}:(h{c['h']},v{c['v']},t{c['delay']},p{100*c['frac']:.1f}%)" for c in comps[:6]])
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["rank", "type", "clear", "patch", "score", "len", "excess", "delay", "min |Δdelay|"],
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.18)
    ax_table.text(
        0.0,
        0.06,
        f"Top true AD components: {top_text}",
        transform=ax_table.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        wrap=True,
    )

    name = f"train_{idx:04d}_paths.png"
    fig.savefig(fig_dir / name, dpi=170)
    plt.close(fig)
    return name, path_rows


def plot_overview(
    fig_dir: Path,
    sample: dict[str, np.ndarray],
    train_pos: np.ndarray,
    bad_ids: np.ndarray,
    scene: np.lib.npyio.NpzFile,
    reflect_count: int,
    args: argparse.Namespace,
) -> str:
    fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)
    wall = (sample["z"] > 2.0) & (np.abs(sample["nz"]) < 0.7)
    ax.scatter(sample["x"], sample["y"], s=0.45, c="#d8d8d8", alpha=0.06, linewidths=0, zorder=0)
    ax.scatter(sample["x"][wall], sample["y"][wall], s=0.75, c="#222222", alpha=0.20, linewidths=0, zorder=1)
    ax.scatter(train_pos[:, 0], train_pos[:, 1], s=5, c="#1f77b4", alpha=0.10, linewidths=0, label="all train", zorder=2)
    ax.scatter(train_pos[bad_ids, 0], train_pos[bad_ids, 1], s=70, facecolors="none", edgecolors="#ff2020", linewidths=1.4, label="bad train", zorder=9)
    for idx in bad_ids:
        rx = train_pos[idx]
        los_blocked = bool(path_clearance(rx, None, scene, args)["blocked"])
        ax.plot(
            [BS_POS[0], rx[0]],
            [BS_POS[1], rx[1]],
            color="#777777" if los_blocked else "#111111",
            lw=0.45,
            ls="--" if los_blocked else "-",
            alpha=0.22,
            zorder=3,
        )
        refl = scene["train_reflection_point"][idx, : args.candidate_pool]
        score = scene["train_reflection_score"][idx, : args.candidate_pool]
        drawn = 0
        for rp, sc in zip(refl, score):
            if drawn >= reflect_count:
                break
            if not np.isfinite(rp[:2]).all():
                continue
            if path_clearance(rx, rp, scene, args)["blocked"]:
                continue
            alpha = float(np.clip(0.15 + 0.55 * sc, 0.12, 0.65))
            ax.plot([BS_POS[0], rp[0], rx[0]], [BS_POS[1], rp[1], rx[1]], color="#e41a1c", lw=0.55, alpha=alpha, zorder=4)
            drawn += 1
    for idx in bad_ids[:30]:
        ax.text(train_pos[idx, 0], train_pos[idx, 1], str(int(idx)), fontsize=7, color="#8a0000", zorder=12, clip_on=True)
    finite_axes(ax, (20, 285), (-220, 110))
    ax.set_title("Occlusion-aware simple ray tracing overview for train LOO bad points")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    name = "train_bad_simple_raytrace_overview.png"
    fig.savefig(fig_dir / name, dpi=190)
    plt.close(fig)
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PATHS.analysis / "visualizations/train_bad_raytrace_occlusion"))
    ap.add_argument("--train-pos", default=str(PATHS.round1 / "Round1_Train_Pos.npy"))
    ap.add_argument("--map", default=str(PATHS.round1 / "Round1_Map.ply"))
    ap.add_argument("--map-sample", default=str(PATHS.analysis / "visualizations/v98_bad_points/data/map_sample_420000_20260708.npz"))
    ap.add_argument("--bad-csv", default=str(PATHS.analysis / "visualizations/current_best_bad_points/data/train_true_bad_top.csv"))
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--truth-support", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--delay-fit", default=ASSETS + "/delay_length_fit.npz")
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--reflect-count", type=int, default=4)
    ap.add_argument("--candidate-pool", type=int, default=8)
    ap.add_argument("--occlusion-samples", type=int, default=220)
    ap.add_argument("--vertical-threshold", type=int, default=2)
    ap.add_argument("--z-margin", type=float, default=0.35)
    ap.add_argument("--span", type=float, default=35.0)
    ap.add_argument("--max-map-points", type=int, default=420000)
    ap.add_argument("--seed", type=int, default=20260708)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_pos = np.load(args.train_pos)
    bad_ids = load_bad_ids(Path(args.bad_csv), args.top_n)
    sample = load_map_sample(Path(args.map), Path(args.map_sample), args.max_map_points, args.seed)
    scene = np.load(args.scene_cache, allow_pickle=True)
    truth_power = np.load(args.truth_support, mmap_mode="r")
    delay_fit = np.load(args.delay_fit)
    delay_slope = float(delay_fit["slope"])
    delay_intercept = float(delay_fit["intercept"])
    geo = np.load(args.geometry_cache)
    regimes = regime_of(geo["train_blocked"].astype(np.float64))

    score_rows: dict[int, dict[str, float | str]] = {}
    with Path(args.bad_csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = int(row["index"])
            score_rows[i] = {
                "score": float(row["score"]),
                "joint": float(row["joint"]),
                "beam": float(row["beam"]),
                "delay": float(row["delay"]),
                "regime": row.get("regime", str(regimes[i])),
            }

    overview = plot_overview(fig_dir, sample, train_pos, bad_ids, scene, args.reflect_count, args)
    all_paths: list[dict[str, float]] = []
    point_figs: list[str] = []
    for idx in bad_ids:
        fig_name, rows = plot_point(
            fig_dir,
            int(idx),
            train_pos[int(idx)],
            score_rows[int(idx)],
            sample,
            scene,
            truth_power,
            delay_slope,
            delay_intercept,
            args.reflect_count,
            args.span,
            args,
        )
        point_figs.append(fig_name)
        all_paths.extend(rows)
        print(f"traced train {idx}: {fig_name}", flush=True)

    csv_path = data_dir / "train_bad_simple_paths.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "point",
            "path_rank",
            "type",
            "patch_id",
            "score",
            "length",
            "excess",
            "delay_est",
            "nearest_true_delay_error",
            "reflect_x",
            "reflect_y",
            "reflect_z",
            "blocked",
            "blocked_bs_ref",
            "blocked_ref_ue",
            "min_clearance",
            "block_hits",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_paths:
            writer.writerow(row)

    # Compact point summary for quick inspection.
    point_summary = []
    for idx in bad_ids:
        rows = [r for r in all_paths if r["point"] == int(idx)]
        refl = [r for r in rows if r["type"] == "one_bounce"]
        clear_rows = [r for r in rows if not bool(r["blocked"])]
        best_delay_err = min([r["nearest_true_delay_error"] for r in rows if np.isfinite(r["nearest_true_delay_error"])], default=float("nan"))
        best_clear_delay_err = min([r["nearest_true_delay_error"] for r in clear_rows if np.isfinite(r["nearest_true_delay_error"])], default=float("nan"))
        clear_refl = [r for r in clear_rows if r["type"] == "one_bounce"]
        point_summary.append(
            {
                "point": int(idx),
                "x": float(train_pos[idx, 0]),
                "y": float(train_pos[idx, 1]),
                "regime": str(score_rows[int(idx)]["regime"]),
                "support_score": float(score_rows[int(idx)]["score"]),
                "best_any_path_delay_error": float(best_delay_err),
                "best_clear_path_delay_error": float(best_clear_delay_err),
                "clear_reflection_count": int(len(clear_refl)),
                "top_reflection_score": float(refl[0]["score"]) if refl else float("nan"),
                "top_reflection_excess": float(refl[0]["excess"]) if refl else float("nan"),
                "figure": f"figures/train_{int(idx):04d}_paths.png",
            }
        )

    summary = {
        "method": "occlusion_aware_train_bad_raytrace",
        "description": "LOS + official-pointcloud one-bounce candidates for train LOO bad points, with segment occlusion filtering",
        "bad_ids": bad_ids.astype(int).tolist(),
        "delay_fit": {"slope": delay_slope, "intercept": delay_intercept},
        "reflect_count": int(args.reflect_count),
        "candidate_pool": int(args.candidate_pool),
        "occlusion": {
            "samples": int(args.occlusion_samples),
            "vertical_threshold": int(args.vertical_threshold),
            "z_margin": float(args.z_margin),
        },
        "overview": f"figures/{overview}",
        "point_summary": point_summary,
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_rows = "\n".join(
        [
            f"| {p['point']} | {p['regime']} | {p['support_score']:.3f} | {p['best_clear_path_delay_error']:.2f} | {p['clear_reflection_count']} | {p['top_reflection_score']:.3f} | {p['top_reflection_excess']:.1f} | [{Path(p['figure']).name}]({p['figure']}) |"
            for p in point_summary
        ]
    )
    md = f"""# Train Bad Point Occlusion-Aware Simple Ray Tracing

This diagnostic traces simple geometric paths for the current worst training points:

- direct LOS path: `BS -> UE`, marked clear or blocked;
- top clear one-bounce specular candidates from a pool of `{args.candidate_pool}` candidates: `BS -> wall point -> UE`;
- all wall patches and reflection candidates are derived from official `Round1_Map.ply`.

It is not a submission and not a full ray tracer.  This version samples every
path segment against the official point-cloud height/wall grid, so candidates
that cross buildings are marked blocked and are not drawn as valid colored rays.

## Overview

- [Global overview](figures/{overview})
- Path table: [train_bad_simple_paths.csv](data/train_bad_simple_paths.csv)

## Per-Point Figures

| train idx | regime | LOO support score | best clear path vs true-delay error | clear reflection count | top raw reflection score | top raw reflection excess / m | figure |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
{md_rows}

## How To Read One Figure

- Solid black line: clear LOS.  Gray dashed line: blocked LOS.
- Colored broken lines: one-bounce reflection candidates that passed segment occlusion checks.
- PDP panel: true training AD delay spectrum; vertical lines show geometric delay estimates from LOS/reflections.
- Beam map: true training AD angular support.
- Table: clear/blocked flag, path length, excess length, delay estimate, and nearest error to the true top AD delays.

If a bad point has clear reflection candidates whose delays are close to strong
true delay peaks, the remaining error is likely component identity/phase/angle
assignment.  If no clear candidate delay is close, the point-cloud geometry
extraction is missing the useful path family or the useful path is multi-bounce
/ diffracted.
"""
    (out_dir / "analysis.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
