#!/usr/bin/env python3
"""Build a stronger continuous scene proxy from the official Round1 point cloud.

The output is not a ray-traced channel.  It is a geometry cache built only from
Round1_Map.ply, train/test positions, and the fixed BS position.  The cache
contains:

* a 2D occupied/height field;
* connected obstacle components from vertical point density;
* split vertical wall patches approximated by rectangular planes;
* LOS height/occupancy features;
* local corridor/ray features around each UE;
* top specular one-bounce candidates from official point-cloud wall patches.

This is intended to replace Sionna mesh/path-token side information with
official-data-only geometry for later support/component models.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from build_map_features import BS_POS, read_ply_memmap


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def percentile_dict(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    return {
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def canonicalize_normals(points_xy: np.ndarray, normals_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normals_xy.astype(np.float64, copy=True)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-12)
    c = np.sum(n * points_xy, axis=1)
    flip = c < 0.0
    n[flip] *= -1.0
    c[flip] *= -1.0
    return n, c


def build_height_grid(vertices, resolution: float, pad: float) -> dict:
    x = np.asarray(vertices["x"], dtype=np.float64)
    y = np.asarray(vertices["y"], dtype=np.float64)
    z = np.asarray(vertices["z"], dtype=np.float64)
    nx_arr = np.asarray(vertices["nx"], dtype=np.float64)
    ny_arr = np.asarray(vertices["ny"], dtype=np.float64)
    nz_arr = np.asarray(vertices["nz"], dtype=np.float64)

    x_min = float(np.floor((x.min() - pad) / resolution) * resolution)
    x_max = float(np.ceil((x.max() + pad) / resolution) * resolution)
    y_min = float(np.floor((y.min() - pad) / resolution) * resolution)
    y_max = float(np.ceil((y.max() + pad) / resolution) * resolution)
    nx_grid = int(np.ceil((x_max - x_min) / resolution)) + 1
    ny_grid = int(np.ceil((y_max - y_min) / resolution)) + 1

    xi = np.floor((x - x_min) / resolution).astype(np.int64)
    yi = np.floor((y - y_min) / resolution).astype(np.int64)
    ok = (xi >= 0) & (xi < nx_grid) & (yi >= 0) & (yi < ny_grid)
    xi = xi[ok]
    yi = yi[ok]
    z_ok = z[ok]
    nx_ok = nx_arr[ok]
    ny_ok = ny_arr[ok]
    nz_ok = nz_arr[ok]
    flat = yi * nx_grid + xi
    size = nx_grid * ny_grid

    count = np.zeros(size, dtype=np.int32)
    z_min = np.full(size, np.inf, dtype=np.float32)
    z_max = np.full(size, -np.inf, dtype=np.float32)
    z_sum = np.zeros(size, dtype=np.float64)
    vertical_count = np.zeros(size, dtype=np.int32)
    horizontal_count = np.zeros(size, dtype=np.int32)
    wall_nx = np.zeros(size, dtype=np.float64)
    wall_ny = np.zeros(size, dtype=np.float64)

    np.add.at(count, flat, 1)
    np.minimum.at(z_min, flat, z_ok.astype(np.float32))
    np.maximum.at(z_max, flat, z_ok.astype(np.float32))
    np.add.at(z_sum, flat, z_ok)

    vertical = (z_ok > 1.0) & (np.abs(nz_ok) < 0.65)
    horizontal = np.abs(nz_ok) > 0.85
    np.add.at(vertical_count, flat[vertical], 1)
    np.add.at(horizontal_count, flat[horizontal], 1)
    np.add.at(wall_nx, flat[vertical], nx_ok[vertical])
    np.add.at(wall_ny, flat[vertical], ny_ok[vertical])

    empty = count == 0
    z_min[empty] = np.nan
    z_max[empty] = np.nan
    z_mean = np.full(size, np.nan, dtype=np.float32)
    z_mean[~empty] = (z_sum[~empty] / count[~empty]).astype(np.float32)
    wall_norm = np.sqrt(wall_nx**2 + wall_ny**2)
    wall_nx_mean = np.zeros(size, dtype=np.float32)
    wall_ny_mean = np.zeros(size, dtype=np.float32)
    valid_n = wall_norm > 1e-8
    wall_nx_mean[valid_n] = (wall_nx[valid_n] / wall_norm[valid_n]).astype(np.float32)
    wall_ny_mean[valid_n] = (wall_ny[valid_n] / wall_norm[valid_n]).astype(np.float32)

    shape = (ny_grid, nx_grid)
    return {
        "x_min": x_min,
        "y_min": y_min,
        "resolution": float(resolution),
        "nx": nx_grid,
        "ny": ny_grid,
        "count": count.reshape(shape),
        "z_min": z_min.reshape(shape),
        "z_max": z_max.reshape(shape),
        "z_mean": z_mean.reshape(shape),
        "vertical_count": vertical_count.reshape(shape),
        "horizontal_count": horizontal_count.reshape(shape),
        "wall_nx": wall_nx_mean.reshape(shape),
        "wall_ny": wall_ny_mean.reshape(shape),
    }


def build_components(vertical_count: np.ndarray, z_max: np.ndarray, min_cell_count: int, min_component_cells: int) -> tuple[np.ndarray, np.ndarray]:
    occupied = (vertical_count >= min_cell_count) & np.isfinite(z_max) & (z_max > 2.0)
    occupied = ndimage.binary_closing(occupied, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labels, n_labels = ndimage.label(occupied, structure=np.ones((3, 3), dtype=np.int8))
    if n_labels == 0:
        return labels.astype(np.int32), np.zeros((0, 8), dtype=np.float32)

    counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
    keep = counts >= min_component_cells
    keep[0] = False
    remap = np.zeros(n_labels + 1, dtype=np.int32)
    remap[np.flatnonzero(keep)] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    labels = remap[labels]

    rows = []
    for label in range(1, int(keep.sum()) + 1):
        yy, xx = np.nonzero(labels == label)
        if yy.size == 0:
            continue
        z_vals = z_max[yy, xx]
        rows.append(
            [
                label,
                yy.size,
                xx.min(),
                xx.max(),
                yy.min(),
                yy.max(),
                np.nanmax(z_vals),
                np.nanmean(z_vals),
            ]
        )
    return labels.astype(np.int32), np.asarray(rows, dtype=np.float32)


def select_vertical_points(vertices, max_abs_nz: float, min_z: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(vertices["x"])
    y = np.asarray(vertices["y"])
    z = np.asarray(vertices["z"])
    nz = np.asarray(vertices["nz"])
    mask = (z >= min_z) & (np.abs(nz) <= max_abs_nz)
    points = np.stack([x[mask], y[mask], z[mask]], axis=1).astype(np.float32)
    normals = np.stack(
        [np.asarray(vertices["nx"])[mask], np.asarray(vertices["ny"])[mask], nz[mask]],
        axis=1,
    ).astype(np.float32)
    return points, normals


def contiguous_intervals(active: np.ndarray) -> list[tuple[int, int]]:
    if not np.any(active):
        return []
    padded = np.concatenate([[False], active, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    return list(zip(starts, ends))


def build_wall_patches(
    wall_points: np.ndarray,
    wall_normals: np.ndarray,
    angle_bin_deg: float,
    offset_bin: float,
    tangent_bin: float,
    min_cluster_count: int,
    min_patch_count: int,
    top_patches: int,
) -> dict:
    points_xy = wall_points[:, :2].astype(np.float64)
    normals_xy = wall_normals[:, :2].astype(np.float64)
    n, c = canonicalize_normals(points_xy, normals_xy)
    theta = np.mod(np.arctan2(n[:, 1], n[:, 0]), 2.0 * np.pi)
    angle_bin = np.floor(theta / np.deg2rad(angle_bin_deg)).astype(np.int64)
    offset_bin_id = np.floor(c / offset_bin).astype(np.int64)
    keys = np.stack([angle_bin, offset_bin_id], axis=1)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)

    rows = []
    for group_id in order:
        count = int(counts[group_id])
        if count < min_cluster_count:
            break
        idx = np.flatnonzero(inv == group_id)
        pts = wall_points[idx].astype(np.float64)
        ns = n[idx]
        cs = c[idx]
        n_mean = np.mean(ns, axis=0)
        n_mean /= max(float(np.linalg.norm(n_mean)), 1e-12)
        c_med = float(np.median(cs))
        tangent = np.array([-n_mean[1], n_mean[0]], dtype=np.float64)
        t = pts[:, :2] @ tangent
        residual = pts[:, :2] @ n_mean - c_med
        good = np.abs(residual) <= max(0.8, np.percentile(np.abs(residual), 90))
        if good.sum() < min_patch_count:
            continue
        t_good = t[good]
        z_good = pts[good, 2]
        res_good = residual[good]
        t_min = np.floor(t_good.min() / tangent_bin) * tangent_bin
        t_bin = np.floor((t_good - t_min) / tangent_bin).astype(np.int64)
        bin_count = np.bincount(t_bin)
        active = bin_count >= 5
        active = ndimage.binary_closing(active, structure=np.ones(3, dtype=bool), iterations=1)
        for start, end in contiguous_intervals(active):
            lo = t_min + start * tangent_bin
            hi = t_min + (end + 1) * tangent_bin
            use = good & (t >= lo) & (t <= hi)
            patch_count = int(use.sum())
            if patch_count < min_patch_count:
                continue
            p = pts[use]
            t_patch = t[use]
            residual_patch = residual[use]
            rows.append(
                [
                    n_mean[0],
                    n_mean[1],
                    c_med,
                    np.percentile(t_patch, 1),
                    np.percentile(t_patch, 99),
                    np.percentile(p[:, 2], 1),
                    np.percentile(p[:, 2], 99),
                    patch_count,
                    np.percentile(np.abs(residual_patch), 50),
                    np.percentile(np.abs(residual_patch), 90),
                    uniq[group_id, 0],
                    uniq[group_id, 1],
                ]
            )

    if not rows:
        arr = np.zeros((0, 12), dtype=np.float32)
    else:
        arr = np.asarray(rows, dtype=np.float64)
        order = np.argsort(-arr[:, 7])
        arr = arr[order[:top_patches]].astype(np.float32)
    return {
        "array": arr,
        "columns": np.asarray(
            [
                "normal_x",
                "normal_y",
                "c",
                "t_min",
                "t_max",
                "z_min",
                "z_max",
                "count",
                "residual_p50",
                "residual_p90",
                "angle_bin",
                "offset_bin",
            ]
        ),
    }


def grid_lookup(grid: np.ndarray, x: np.ndarray, y: np.ndarray, meta: dict, fill: float = 0.0) -> np.ndarray:
    xi = np.floor((x - meta["x_min"]) / meta["resolution"]).astype(np.int64)
    yi = np.floor((y - meta["y_min"]) / meta["resolution"]).astype(np.int64)
    valid = (xi >= 0) & (xi < meta["nx"]) & (yi >= 0) & (yi < meta["ny"])
    out = np.full(x.shape, fill, dtype=np.float32)
    out[valid] = grid[yi[valid], xi[valid]]
    return out


def query_los_height_features(pos: np.ndarray, grid: dict, samples: np.ndarray, batch: int) -> tuple[np.ndarray, list[str]]:
    z_max = grid["z_max"]
    vertical = grid["vertical_count"].astype(np.float32)
    names = [
        "pc_los_height_block_frac",
        "pc_los_height_block_count",
        "pc_los_height_surplus_sum",
        "pc_los_height_surplus_max",
        "pc_los_height_first_t",
        "pc_los_height_last_t",
        "pc_los_vertical_density_mean",
        "pc_los_vertical_density_max",
        "pc_los_roof_over_line_frac",
        "pc_los_near_bs_block_frac",
        "pc_los_near_ue_block_frac",
    ]
    rows = []
    for start in range(0, len(pos), batch):
        end = min(start + batch, len(pos))
        ue = pos[start:end].astype(np.float64)
        line = BS_POS[None, None, :] * (1.0 - samples[None, :, None]) + ue[:, None, :] * samples[None, :, None]
        x = line[..., 0].reshape(-1)
        y = line[..., 1].reshape(-1)
        line_z = line[..., 2].reshape(end - start, samples.size)
        h = grid_lookup(z_max, x, y, grid, fill=np.nan).reshape(end - start, samples.size)
        v = grid_lookup(vertical, x, y, grid, fill=0.0).reshape(end - start, samples.size)
        surplus = np.maximum(h - line_z, 0.0)
        surplus = np.nan_to_num(surplus, nan=0.0, posinf=0.0, neginf=0.0)
        block = np.isfinite(h) & (v >= 1.0) & (surplus > 0.25)
        roof_over = np.isfinite(h) & (surplus > 2.0)
        any_block = block.any(axis=1)
        first = np.ones(end - start, dtype=np.float64)
        last = np.zeros(end - start, dtype=np.float64)
        if np.any(any_block):
            first_idx = np.argmax(block[any_block], axis=1)
            last_idx = block.shape[1] - 1 - np.argmax(block[any_block, ::-1], axis=1)
            first[any_block] = samples[first_idx]
            last[any_block] = samples[last_idx]
        near_bs = samples < 0.35
        near_ue = samples > 0.65
        feat = np.stack(
            [
                block.mean(axis=1),
                block.sum(axis=1),
                surplus.sum(axis=1) / samples.size,
                np.max(surplus, axis=1),
                first,
                last,
                v.mean(axis=1),
                v.max(axis=1),
                roof_over.mean(axis=1),
                block[:, near_bs].mean(axis=1),
                block[:, near_ue].mean(axis=1),
            ],
            axis=1,
        )
        rows.append(feat.astype(np.float32))
    return np.concatenate(rows, axis=0), names


def query_pointcloud_line_features(
    wall_tree: cKDTree,
    wall_points: np.ndarray,
    wall_normals: np.ndarray,
    pos: np.ndarray,
    samples: np.ndarray,
    thresholds: list[float],
    batch: int,
) -> tuple[np.ndarray, list[str]]:
    names = [
        "pc_los_nn_min_dist",
        "pc_los_nn_p05_dist",
        "pc_los_nn_p10_dist",
        "pc_los_nn_mean_inv",
        "pc_los_nn_nearest_t",
        "pc_los_nn_nearest_nx",
        "pc_los_nn_nearest_ny",
        "pc_los_nn_nearest_nz",
    ]
    for radius in thresholds:
        tag = str(radius).replace(".", "p")
        names.extend([f"pc_los_nn_count_r{tag}", f"pc_los_nn_frac_r{tag}", f"pc_los_nn_first_t_r{tag}", f"pc_los_nn_last_t_r{tag}"])
    rows = []
    eps = 1e-6
    for start in range(0, len(pos), batch):
        end = min(start + batch, len(pos))
        ue = pos[start:end].astype(np.float64)
        line = BS_POS[None, None, :] * (1.0 - samples[None, :, None]) + ue[:, None, :] * samples[None, :, None]
        q = line.reshape(-1, 3)
        dist, idx = wall_tree.query(q, k=1, workers=-1)
        dist = dist.reshape(end - start, samples.size)
        idx = idx.reshape(end - start, samples.size)
        nearest_j = np.argmin(dist, axis=1)
        nearest_idx = idx[np.arange(end - start), nearest_j]
        nearest_normals = wall_normals[nearest_idx]
        feats = [
            np.min(dist, axis=1),
            np.percentile(dist, 5, axis=1),
            np.percentile(dist, 10, axis=1),
            np.mean(1.0 / np.maximum(dist, eps), axis=1),
            samples[nearest_j],
            nearest_normals[:, 0],
            nearest_normals[:, 1],
            nearest_normals[:, 2],
        ]
        for radius in thresholds:
            hit = dist < radius
            any_hit = hit.any(axis=1)
            first = np.ones(end - start, dtype=np.float64)
            last = np.zeros(end - start, dtype=np.float64)
            if np.any(any_hit):
                first_idx = np.argmax(hit[any_hit], axis=1)
                last_idx = hit.shape[1] - 1 - np.argmax(hit[any_hit, ::-1], axis=1)
                first[any_hit] = samples[first_idx]
                last[any_hit] = samples[last_idx]
            feats.extend([hit.sum(axis=1), hit.mean(axis=1), first, last])
        rows.append(np.stack(feats, axis=1).astype(np.float32))
    return np.concatenate(rows, axis=0), names


def query_local_corridor_features(pos: np.ndarray, grid: dict, angles: np.ndarray, max_dist: float) -> tuple[np.ndarray, list[str]]:
    step = grid["resolution"]
    steps = np.arange(step, max_dist + 1e-6, step, dtype=np.float64)
    occupied = (grid["vertical_count"] >= 1) & np.isfinite(grid["z_max"]) & (grid["z_max"] > 2.0)
    all_dist = np.full((len(pos), len(angles)), max_dist, dtype=np.float32)
    for ai, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        hit_done = np.zeros(len(pos), dtype=bool)
        for dist in steps:
            qxy = pos[:, :2] + direction[None, :] * dist
            xi = np.floor((qxy[:, 0] - grid["x_min"]) / grid["resolution"]).astype(np.int64)
            yi = np.floor((qxy[:, 1] - grid["y_min"]) / grid["resolution"]).astype(np.int64)
            valid = (xi >= 0) & (xi < grid["nx"]) & (yi >= 0) & (yi < grid["ny"])
            hit = np.zeros(len(pos), dtype=bool)
            hit[valid] = occupied[yi[valid], xi[valid]]
            new_hit = hit & ~hit_done
            all_dist[new_hit, ai] = dist
            hit_done |= hit
            if hit_done.all():
                break
    names = []
    for ai in range(len(angles)):
        names.append(f"pc_local_ray_dist_{ai:02d}")
    summary = np.stack(
        [
            all_dist.min(axis=1),
            np.percentile(all_dist, 10, axis=1),
            np.median(all_dist, axis=1),
            all_dist.mean(axis=1),
            all_dist.max(axis=1),
            all_dist.std(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    names.extend(
        [
            "pc_local_ray_min",
            "pc_local_ray_p10",
            "pc_local_ray_median",
            "pc_local_ray_mean",
            "pc_local_ray_max",
            "pc_local_ray_std",
        ]
    )
    return np.concatenate([all_dist, summary], axis=1), names


def build_reflection_candidates(pos: np.ndarray, patches: np.ndarray, top_k: int, batch: int) -> dict:
    if patches.size == 0:
        empty_i = np.full((len(pos), top_k), -1, dtype=np.int32)
        empty_f = np.zeros((len(pos), top_k), dtype=np.float32)
        return {
            "patch_id": empty_i,
            "score": empty_f,
            "path_length": empty_f,
            "excess_length": empty_f,
            "reflection_point": np.zeros((len(pos), top_k, 3), dtype=np.float32),
        }

    normals = patches[:, :2].astype(np.float64)
    c = patches[:, 2].astype(np.float64)
    t_min = patches[:, 3].astype(np.float64)
    t_max = patches[:, 4].astype(np.float64)
    z_min = patches[:, 5].astype(np.float64)
    z_max = patches[:, 6].astype(np.float64)
    count = patches[:, 7].astype(np.float64)
    residual = patches[:, 9].astype(np.float64)
    tangent = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
    bs_xy = BS_POS[:2].astype(np.float64)
    bs_z = float(BS_POS[2])
    signed = normals @ bs_xy - c
    mirror_bs = bs_xy[None, :] - 2.0 * signed[:, None] * normals
    mirror_z = bs_z

    out_id = np.full((len(pos), top_k), -1, dtype=np.int32)
    out_score = np.zeros((len(pos), top_k), dtype=np.float32)
    out_path = np.zeros((len(pos), top_k), dtype=np.float32)
    out_excess = np.zeros((len(pos), top_k), dtype=np.float32)
    out_refl = np.zeros((len(pos), top_k, 3), dtype=np.float32)
    count_scale = np.log1p(count) / max(float(np.log1p(count.max())), 1e-6)

    for start in range(0, len(pos), batch):
        end = min(start + batch, len(pos))
        ue = pos[start:end].astype(np.float64)
        ue_xy = ue[:, :2]
        ue_z = ue[:, 2]
        diff = ue_xy[:, None, :] - mirror_bs[None, :, :]
        denom = np.einsum("bpj,pj->bp", diff, normals)
        numer = c[None, :] - np.einsum("pj,pj->p", mirror_bs, normals)[None, :]
        t = numer / np.where(np.abs(denom) < 1e-9, np.nan, denom)
        refl_xy = mirror_bs[None, :, :] + t[..., None] * diff
        refl_z = mirror_z + t * (ue_z[:, None] - mirror_z)
        tang = np.einsum("bpj,pj->bp", refl_xy, tangent)
        inside_t = (tang >= (t_min[None, :] - 1.5)) & (tang <= (t_max[None, :] + 1.5))
        inside_z = (refl_z >= (z_min[None, :] - 1.0)) & (refl_z <= (z_max[None, :] + 1.0))
        valid = (t > 0.0) & (t < 1.0) & inside_t & inside_z

        refl3 = np.concatenate([refl_xy, refl_z[..., None]], axis=2)
        d1 = np.linalg.norm(refl3 - BS_POS[None, None, :], axis=2)
        d2 = np.linalg.norm(refl3 - ue[:, None, :], axis=2)
        path_len = d1 + d2
        direct = np.linalg.norm(ue - BS_POS[None, :], axis=1)
        excess = path_len - direct[:, None]
        edge_margin = np.minimum(tang - t_min[None, :], t_max[None, :] - tang)
        z_margin = np.minimum(refl_z - z_min[None, :], z_max[None, :] - refl_z)
        edge_gate = 1.0 / (1.0 + np.exp(-np.clip(edge_margin - 0.5, -60.0, 60.0)))
        z_gate = 1.0 / (1.0 + np.exp(-np.clip(z_margin - 0.5, -60.0, 60.0)))
        score = (
            valid.astype(np.float64)
            * count_scale[None, :]
            * edge_gate
            * z_gate
            * np.exp(-np.maximum(excess, 0.0) / 260.0)
            / (1.0 + residual[None, :])
        )
        if score.shape[1] <= top_k:
            top_idx = np.argsort(-score, axis=1)[:, :top_k]
        else:
            part = np.argpartition(-score, kth=top_k - 1, axis=1)[:, :top_k]
            row = np.arange(score.shape[0])[:, None]
            order = np.argsort(-score[row, part], axis=1)
            top_idx = part[row, order]
        row = np.arange(end - start)[:, None]
        out_id[start:end] = top_idx.astype(np.int32)
        out_score[start:end] = score[row, top_idx].astype(np.float32)
        out_path[start:end] = path_len[row, top_idx].astype(np.float32)
        out_excess[start:end] = excess[row, top_idx].astype(np.float32)
        out_refl[start:end] = refl3[row, top_idx].astype(np.float32)

    return {
        "patch_id": out_id,
        "score": out_score,
        "path_length": out_path,
        "excess_length": out_excess,
        "reflection_point": out_refl,
    }


def write_patch_obj(path: Path, patches: np.ndarray, max_patches: int) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        f.write("# simplified vertical wall patches from official Round1_Map.ply\n")
        vertex_id = 1
        for patch in patches[:max_patches]:
            n = patch[:2].astype(np.float64)
            n /= max(float(np.linalg.norm(n)), 1e-12)
            tangent = np.array([-n[1], n[0]], dtype=np.float64)
            c = float(patch[2])
            t0, t1 = float(patch[3]), float(patch[4])
            z0, z1 = float(patch[5]), float(patch[6])
            corners = []
            for tt, zz in [(t0, z0), (t1, z0), (t1, z1), (t0, z1)]:
                xy = c * n + tt * tangent
                corners.append((xy[0], xy[1], zz))
            for x, y, z in corners:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            f.write(f"f {vertex_id} {vertex_id + 1} {vertex_id + 2} {vertex_id + 3}\n")
            vertex_id += 4


def save_visualizations(out_dir: Path, grid: dict, patches: np.ndarray, train_pos: np.ndarray, test_pos: np.ndarray) -> list[str]:
    ensure_dir(out_dir)
    paths = []
    extent = [
        grid["x_min"],
        grid["x_min"] + grid["resolution"] * grid["nx"],
        grid["y_min"],
        grid["y_min"] + grid["resolution"] * grid["ny"],
    ]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    z = np.nan_to_num(grid["z_max"], nan=0.0)
    im = ax.imshow(z, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=np.nanpercentile(z, 99))
    ax.scatter(train_pos[:, 0], train_pos[:, 1], s=2, c="#ffffff", alpha=0.35, label="train")
    ax.scatter(test_pos[:, 0], test_pos[:, 1], s=5, c="#ff3b30", alpha=0.75, label="test")
    ax.scatter([BS_POS[0]], [BS_POS[1]], s=70, c="#00d1ff", marker="*", label="BS")
    ax.set_title("Official point cloud height field")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="z_max / m")
    fig.tight_layout()
    p = out_dir / "height_field_train_test.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    vc = np.log1p(grid["vertical_count"])
    im = ax.imshow(vc, origin="lower", extent=extent, cmap="magma")
    for patch in patches[:250]:
        n = patch[:2].astype(np.float64)
        n /= max(float(np.linalg.norm(n)), 1e-12)
        tangent = np.array([-n[1], n[0]], dtype=np.float64)
        c = float(patch[2])
        t0, t1 = float(patch[3]), float(patch[4])
        p0 = c * n + t0 * tangent
        p1 = c * n + t1 * tangent
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#00e5ff", lw=0.55, alpha=0.65)
    ax.scatter(test_pos[:, 0], test_pos[:, 1], s=4, c="#ffffff", alpha=0.75)
    ax.scatter([BS_POS[0]], [BS_POS[1]], s=70, c="#00d1ff", marker="*")
    ax.set_title("Vertical wall density and extracted wall patches")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    fig.colorbar(im, ax=ax, label="log(1+vertical_count)")
    fig.tight_layout()
    p = out_dir / "wall_patches_topdown.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(str(p))

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.artifacts / "shared_features/official_pointcloud_scene_cache.npz"))
    parser.add_argument("--summary", default=str(PATHS.analysis / "results/json/official_pointcloud_scene_summary.json"))
    parser.add_argument("--obj", default=str(PATHS.artifacts / "shared_features/official_pointcloud_wall_patches.obj"))
    parser.add_argument("--viz-dir", default=str(PATHS.analysis / "visualizations/official_pointcloud_scene"))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--grid-pad", type=float, default=2.0)
    parser.add_argument("--line-samples", type=int, default=160)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-abs-nz", type=float, default=0.65)
    parser.add_argument("--min-wall-z", type=float, default=1.0)
    parser.add_argument("--angle-bin-deg", type=float, default=2.0)
    parser.add_argument("--offset-bin", type=float, default=0.75)
    parser.add_argument("--tangent-bin", type=float, default=1.0)
    parser.add_argument("--min-cluster-count", type=int, default=220)
    parser.add_argument("--min-patch-count", type=int, default=80)
    parser.add_argument("--top-patches", type=int, default=1600)
    parser.add_argument("--candidate-patches", type=int, default=768)
    parser.add_argument("--reflection-top-k", type=int, default=8)
    parser.add_argument("--corridor-rays", type=int, default=16)
    parser.add_argument("--corridor-max-dist", type=float, default=80.0)
    args = parser.parse_args()

    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    vertices = read_ply_memmap(data / "Round1_Map.ply")
    print(f"loaded official point cloud vertices={len(vertices)}", flush=True)

    grid = build_height_grid(vertices, args.resolution, args.grid_pad)
    labels, component_table = build_components(
        grid["vertical_count"], grid["z_max"], min_cell_count=2, min_component_cells=8
    )
    wall_points, wall_normals = select_vertical_points(vertices, args.max_abs_nz, args.min_wall_z)
    print(f"selected vertical wall points={len(wall_points)}", flush=True)
    wall_patches = build_wall_patches(
        wall_points,
        wall_normals,
        args.angle_bin_deg,
        args.offset_bin,
        args.tangent_bin,
        args.min_cluster_count,
        args.min_patch_count,
        args.top_patches,
    )
    patches = wall_patches["array"]
    print(f"built wall patches={len(patches)}", flush=True)

    wall_tree = cKDTree(wall_points)
    line_samples = np.linspace(0.02, 0.98, args.line_samples, dtype=np.float64)
    nn_thresholds = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    train_los_h, names_los_h = query_los_height_features(train_pos, grid, line_samples, args.batch)
    test_los_h, _ = query_los_height_features(test_pos, grid, line_samples, args.batch)
    train_los_nn, names_los_nn = query_pointcloud_line_features(
        wall_tree, wall_points, wall_normals, train_pos, line_samples, nn_thresholds, args.batch
    )
    test_los_nn, _ = query_pointcloud_line_features(
        wall_tree, wall_points, wall_normals, test_pos, line_samples, nn_thresholds, args.batch
    )
    angles = np.linspace(0.0, 2.0 * np.pi, args.corridor_rays, endpoint=False, dtype=np.float64)
    train_corridor, names_corridor = query_local_corridor_features(train_pos, grid, angles, args.corridor_max_dist)
    test_corridor, _ = query_local_corridor_features(test_pos, grid, angles, args.corridor_max_dist)

    train_features = np.concatenate([train_los_h, train_los_nn, train_corridor], axis=1).astype(np.float32)
    test_features = np.concatenate([test_los_h, test_los_nn, test_corridor], axis=1).astype(np.float32)
    feature_names = np.asarray(names_los_h + names_los_nn + names_corridor)

    candidate_patches = patches[: min(args.candidate_patches, len(patches))]
    train_refl = build_reflection_candidates(train_pos, candidate_patches, args.reflection_top_k, args.batch)
    test_refl = build_reflection_candidates(test_pos, candidate_patches, args.reflection_top_k, args.batch)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        x_min=np.asarray(grid["x_min"], dtype=np.float32),
        y_min=np.asarray(grid["y_min"], dtype=np.float32),
        resolution=np.asarray(grid["resolution"], dtype=np.float32),
        grid_nx=np.asarray(grid["nx"], dtype=np.int32),
        grid_ny=np.asarray(grid["ny"], dtype=np.int32),
        grid_count=grid["count"],
        grid_z_min=grid["z_min"],
        grid_z_max=grid["z_max"],
        grid_z_mean=grid["z_mean"],
        grid_vertical_count=grid["vertical_count"],
        grid_horizontal_count=grid["horizontal_count"],
        grid_wall_nx=grid["wall_nx"],
        grid_wall_ny=grid["wall_ny"],
        component_labels=labels,
        component_table=component_table,
        component_columns=np.asarray(["label", "cell_count", "x0", "x1", "y0", "y1", "z_max", "z_mean"]),
        wall_patches=patches,
        wall_patch_columns=wall_patches["columns"],
        feature_names=feature_names,
        train_features=train_features,
        test_features=test_features,
        train_reflection_patch_id=train_refl["patch_id"],
        train_reflection_score=train_refl["score"],
        train_reflection_path_length=train_refl["path_length"],
        train_reflection_excess_length=train_refl["excess_length"],
        train_reflection_point=train_refl["reflection_point"],
        test_reflection_patch_id=test_refl["patch_id"],
        test_reflection_score=test_refl["score"],
        test_reflection_path_length=test_refl["path_length"],
        test_reflection_excess_length=test_refl["excess_length"],
        test_reflection_point=test_refl["reflection_point"],
        line_samples=line_samples.astype(np.float32),
        corridor_angles=angles.astype(np.float32),
    )

    obj_path = Path(args.obj)
    write_patch_obj(obj_path, patches, max_patches=min(1200, len(patches)))
    viz_paths = save_visualizations(Path(args.viz_dir), grid, patches, train_pos, test_pos)

    summary = {
        "source": str(data / "Round1_Map.ply"),
        "cache": str(out_path),
        "obj": str(obj_path),
        "analysis/visualizations": viz_paths,
        "vertex_count": int(len(vertices)),
        "grid": {
            "resolution": float(args.resolution),
            "nx": int(grid["nx"]),
            "ny": int(grid["ny"]),
            "x_min": float(grid["x_min"]),
            "y_min": float(grid["y_min"]),
            "occupied_cells": int(np.isfinite(grid["z_max"]).sum()),
            "vertical_cells": int(((grid["vertical_count"] >= 1) & np.isfinite(grid["z_max"])).sum()),
        },
        "component_count": int(component_table.shape[0]),
        "wall_points": int(len(wall_points)),
        "wall_patch_count": int(len(patches)),
        "wall_patch_top_counts": patches[:10, 7].astype(int).tolist() if len(patches) else [],
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names.tolist(),
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "train_los_block_frac": percentile_dict(train_features[:, feature_names.tolist().index("pc_los_height_block_frac")]),
        "test_los_block_frac": percentile_dict(test_features[:, feature_names.tolist().index("pc_los_height_block_frac")]),
        "train_reflection_top_score": percentile_dict(train_refl["score"][:, 0]),
        "test_reflection_top_score": percentile_dict(test_refl["score"][:, 0]),
        "train_reflection_top_excess": percentile_dict(train_refl["excess_length"][:, 0]),
        "test_reflection_top_excess": percentile_dict(test_refl["excess_length"][:, 0]),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "cache": str(out_path),
        "summary": args.summary,
        "obj": str(obj_path),
        "wall_patches": int(len(patches)),
        "features": list(train_features.shape),
        "analysis/visualizations": viz_paths,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
