#!/usr/bin/env python3
"""Build UE-plane-consistent wall ribbons from the official point-cloud grid."""

from __future__ import annotations

import json

import numpy as np
from project_paths import PATHS


VERSION_NUMBER = 272
VERSION = "v272_extruded_pointcloud_scene"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
SCENE_DIR = OUT_DIR / "scene"
SCENE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MESH = SCENE_DIR / "extruded_buildings.obj"
FOOTPRINT = OUT_DIR / "v272_extruded_footprint.npz"
SUMMARY = REPORT_DIR / "v272_extruded_mesh_build.json"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"


def add_quad(handle, corners: list[tuple[float, float, float]], vertex_id: int) -> int:
    for x, y, z in corners:
        handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
    handle.write(
        f"f {vertex_id} {vertex_id + 1} {vertex_id + 2} {vertex_id + 3}\n"
    )
    return vertex_id + 4


def main() -> None:
    if not SCENE_CACHE.is_file():
        raise FileNotFoundError(SCENE_CACHE)
    scene = np.load(SCENE_CACHE)
    labels = scene["component_labels"].astype(np.int32)
    z_grid = scene["grid_z_max"].astype(np.float32)
    table = scene["component_table"].astype(np.float32)
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    all_pos = np.concatenate([train_pos, test_pos], axis=0)
    ue_clear_grid = np.zeros_like(labels, dtype=bool)
    ue_x = np.floor((all_pos[:, 0] - x_min) / resolution).astype(np.int64)
    ue_y = np.floor((all_pos[:, 1] - y_min) / resolution).astype(np.int64)
    ue_inside = (
        (ue_x >= 0)
        & (ue_x < labels.shape[1])
        & (ue_y >= 0)
        & (ue_y < labels.shape[0])
    )
    ue_clear_grid[ue_y[ue_inside], ue_x[ue_inside]] = True
    wall_conflict_count = int(np.sum((labels > 0) & ue_clear_grid))
    summaries = []
    total_cells = 0
    total_boundary_edges = 0
    footprint_grid = np.zeros_like(labels, dtype=bool)
    vertex_id = 1
    with MESH.open("w", encoding="utf-8") as handle:
        handle.write("# extruded wall ribbons reconstructed from official Round1_Map.ply\n")
        for row in table:
            label = int(row[0])
            component = labels == label
            if not np.any(component):
                continue
            ys, xs = np.where(component)
            pad = 4
            y0 = max(int(ys.min()) - pad, 0)
            y1 = min(int(ys.max()) + pad + 1, labels.shape[0])
            x0 = max(int(xs.min()) - pad, 0)
            x1 = min(int(xs.max()) + pad + 1, labels.shape[1])
            local_mask = component[y0:y1, x0:x1]
            footprint = local_mask & ~ue_clear_grid[y0:y1, x0:x1]
            footprint_grid[y0:y1, x0:x1] |= footprint
            wall_heights = z_grid[component & np.isfinite(z_grid)]
            if len(wall_heights) == 0:
                height = max(float(row[6]), 3.0)
            else:
                height = max(float(np.quantile(wall_heights, 0.82)), 3.0)
            cell_count = int(np.sum(footprint))
            boundary_edges = 0
            for ly, lx in zip(*np.where(footprint)):
                gx = x0 + int(lx)
                gy = y0 + int(ly)
                xa = x_min + gx * resolution
                xb = xa + resolution
                ya = y_min + gy * resolution
                yb = ya + resolution
                vertex_id = add_quad(
                    handle,
                    [(xa, ya, height), (xb, ya, height), (xb, yb, height), (xa, yb, height)],
                    vertex_id,
                )
                for dy, dx, corners in (
                    (-1, 0, [(xa, ya, 0.0), (xb, ya, 0.0), (xb, ya, height), (xa, ya, height)]),
                    (1, 0, [(xb, yb, 0.0), (xa, yb, 0.0), (xa, yb, height), (xb, yb, height)]),
                    (0, -1, [(xa, yb, 0.0), (xa, ya, 0.0), (xa, ya, height), (xa, yb, height)]),
                    (0, 1, [(xb, ya, 0.0), (xb, yb, 0.0), (xb, yb, height), (xb, ya, height)]),
                ):
                    ny = ly + dy
                    nx = lx + dx
                    outside = (
                        ny < 0
                        or ny >= footprint.shape[0]
                        or nx < 0
                        or nx >= footprint.shape[1]
                        or not footprint[ny, nx]
                    )
                    if outside:
                        vertex_id = add_quad(handle, corners, vertex_id)
                        boundary_edges += 1
            total_cells += cell_count
            total_boundary_edges += boundary_edges
            summaries.append(
                {
                    "label": label,
                    "wall_cells": int(np.sum(component)),
                    "footprint_cells": cell_count,
                    "fill_ratio": float(cell_count / max(int(np.sum(component)), 1)),
                    "height": height,
                    "boundary_edges": boundary_edges,
                }
            )
    def point_occupancy(pos: np.ndarray) -> np.ndarray:
        xi = np.floor((pos[:, 0] - x_min) / resolution).astype(np.int64)
        yi = np.floor((pos[:, 1] - y_min) / resolution).astype(np.int64)
        inside = (xi >= 0) & (xi < labels.shape[1]) & (yi >= 0) & (yi < labels.shape[0])
        occupied = np.zeros(len(pos), dtype=bool)
        occupied[inside] = footprint_grid[yi[inside], xi[inside]]
        return occupied

    train_occupied = point_occupancy(train_pos)
    test_occupied = point_occupancy(test_pos)
    np.savez_compressed(
        FOOTPRINT,
        footprint=footprint_grid,
        x_min=np.asarray(x_min, dtype=np.float32),
        y_min=np.asarray(y_min, dtype=np.float32),
        resolution=np.asarray(resolution, dtype=np.float32),
        train_occupied=train_occupied,
        test_occupied=test_occupied,
    )
    report = {
        "method": VERSION,
        "status": "mesh_built",
        "source": PATHS.relative(SCENE_CACHE),
        "mesh": PATHS.relative(MESH),
        "footprint": PATHS.relative(FOOTPRINT),
        "grid": {
            "shape": list(labels.shape),
            "resolution": resolution,
            "component_count": len(summaries),
            "mode": "ue_plane_consistent_wall_ribbons",
            "removed_ue_conflict_cells": wall_conflict_count,
        },
        "mesh_stats": {
            "footprint_cells": total_cells,
            "boundary_edges": total_boundary_edges,
            "quad_count": total_cells + total_boundary_edges,
            "vertex_count": 4 * (total_cells + total_boundary_edges),
        },
        "ue_plane_audit": {
            "train_z_unique": np.unique(train_pos[:, 2]).tolist(),
            "test_z_unique": np.unique(test_pos[:, 2]).tolist(),
            "train_inside_footprint": int(np.sum(train_occupied)),
            "test_inside_footprint": int(np.sum(test_occupied)),
            "train_inside_fraction": float(np.mean(train_occupied)),
            "test_inside_fraction": float(np.mean(test_occupied)),
        },
        "components": summaries,
    }
    SUMMARY.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
