#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
from pathlib import Path

import numpy as np

from build_map_features import read_ply_memmap, select_wall_points


def canonical_normals(normals_xy, points_xy):
    n = normals_xy.astype(np.float64, copy=True)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-12)
    c = np.sum(n * points_xy, axis=1)
    flip = c < 0
    n[flip] *= -1.0
    c[flip] *= -1.0
    return n, c


def build_planes(points, normals, angle_bin_deg, offset_bin, min_count, top_count):
    points_xy = points[:, :2].astype(np.float64)
    normals_xy = normals[:, :2].astype(np.float64)
    n, c = canonical_normals(normals_xy, points_xy)
    theta = np.mod(np.arctan2(n[:, 1], n[:, 0]), 2.0 * np.pi)
    angle_bin = np.floor(theta / np.deg2rad(angle_bin_deg)).astype(np.int64)
    offset_bin_id = np.floor(c / offset_bin).astype(np.int64)
    keys = np.stack([angle_bin, offset_bin_id], axis=1)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)

    planes = []
    for group_id in order:
        count = int(counts[group_id])
        if count < min_count:
            break
        idx = np.flatnonzero(inv == group_id)
        pts = points[idx].astype(np.float64)
        ns = n[idx]
        cs = c[idx]
        n_mean = np.mean(ns, axis=0)
        n_mean /= max(float(np.linalg.norm(n_mean)), 1e-12)
        c_mean = float(np.median(cs))
        # Recompute tangent coordinates with the final plane normal.
        tangent = np.array([-n_mean[1], n_mean[0]], dtype=np.float64)
        t_coord = pts[:, :2] @ tangent
        residual = pts[:, :2] @ n_mean - c_mean
        plane = {
            "normal": n_mean.astype(float).tolist(),
            "c": c_mean,
            "count": count,
            "t_min": float(np.percentile(t_coord, 1)),
            "t_max": float(np.percentile(t_coord, 99)),
            "z_min": float(np.percentile(pts[:, 2], 1)),
            "z_max": float(np.percentile(pts[:, 2], 99)),
            "residual_p50": float(np.percentile(np.abs(residual), 50)),
            "residual_p90": float(np.percentile(np.abs(residual), 90)),
        }
        planes.append(plane)
        if len(planes) >= top_count:
            break
    return planes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/wall_planes.json"))
    parser.add_argument("--x-min", type=float, default=-50.0)
    parser.add_argument("--x-max", type=float, default=300.0)
    parser.add_argument("--y-min", type=float, default=-250.0)
    parser.add_argument("--y-max", type=float, default=150.0)
    parser.add_argument("--angle-bin-deg", type=float, default=3.0)
    parser.add_argument("--offset-bin", type=float, default=1.0)
    parser.add_argument("--min-count", type=int, default=400)
    parser.add_argument("--top-count", type=int, default=200)
    args = parser.parse_args()

    data = Path(args.data)
    vertices = read_ply_memmap(data / "Round1_Map.ply")
    points, normals = select_wall_points(vertices, args.x_min, args.x_max, args.y_min, args.y_max)
    planes = build_planes(points, normals, args.angle_bin_deg, args.offset_bin, args.min_count, args.top_count)
    payload = {
        "source": str(data / "Round1_Map.ply"),
        "wall_points": int(len(points)),
        "angle_bin_deg": float(args.angle_bin_deg),
        "offset_bin": float(args.offset_bin),
        "min_count": int(args.min_count),
        "planes": planes,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({
        "out": args.out,
        "wall_points": int(len(points)),
        "plane_count": len(planes),
        "top_counts": [p["count"] for p in planes[:10]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
