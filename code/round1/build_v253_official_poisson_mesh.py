#!/usr/bin/env python3
"""v253: reconstruct a closed scene mesh from the official Round1 point cloud."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import open3d as o3d

from project_paths import PATHS


VERSION_NUMBER = 253
VERSION = "v253_official_poisson_mesh"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
SCENE_DIR = OUT_DIR / "scene"
MESH = SCENE_DIR / "official_poisson_mesh.ply"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxel-size", type=float, default=0.45)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.03)
    parser.add_argument("--density-quantile", type=float, default=0.018)
    parser.add_argument("--target-triangles", type=int, default=150_000)
    return parser.parse_args()


def mesh_summary(mesh: o3d.geometry.TriangleMesh) -> dict[str, object]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "bbox_min": vertices.min(axis=0).astype(float).tolist() if len(vertices) else [],
        "bbox_max": vertices.max(axis=0).astype(float).tolist() if len(vertices) else [],
        "edge_manifold": bool(mesh.is_edge_manifold(allow_boundary_edges=True)),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "watertight": bool(mesh.is_watertight()),
        "orientable": bool(mesh.is_orientable()),
    }


def main() -> None:
    args = parse_args()
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source = PATHS.round1 / "Round1_Map.ply"
    if not source.is_file():
        raise FileNotFoundError(source)

    started = time.time()
    pcd = o3d.io.read_point_cloud(str(source))
    if len(pcd.points) == 0 or not pcd.has_normals():
        raise ValueError("Official point cloud is empty or has no normals")
    original_count = len(pcd.points)
    bounds = pcd.get_axis_aligned_bounding_box()
    pcd = pcd.voxel_down_sample(args.voxel_size)
    pcd.normalize_normals()
    downsampled_count = len(pcd.points)
    print(
        f"v253 point cloud original={original_count} downsampled={downsampled_count}",
        flush=True,
    )

    t0 = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=args.depth,
        scale=args.scale,
        linear_fit=True,
        n_threads=-1,
    )
    poisson_seconds = time.time() - t0
    densities_np = np.asarray(densities)
    threshold = float(np.quantile(densities_np, args.density_quantile))
    mesh.remove_vertices_by_mask(densities_np < threshold)
    crop = o3d.geometry.AxisAlignedBoundingBox(
        bounds.min_bound - np.asarray([1.0, 1.0, 0.6]),
        bounds.max_bound + np.asarray([1.0, 1.0, 0.6]),
    )
    mesh = mesh.crop(crop)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    before_simplify = mesh_summary(mesh)

    if len(mesh.triangles) > args.target_triangles:
        mesh = mesh.simplify_quadric_decimation(args.target_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    after_simplify = mesh_summary(mesh)
    if not o3d.io.write_triangle_mesh(
        str(MESH), mesh, write_ascii=False, compressed=False, write_vertex_normals=True
    ):
        raise IOError(f"Failed to write {MESH}")

    report = {
        "method": VERSION,
        "status": "geometry_cache_only",
        "primary_hypothesis": (
            "A closed Poisson mesh preserves official point-cloud topology and "
            "visibility better than disconnected rectangular wall patches."
        ),
        "compliance_boundary": {
            "uses": ["official Round1_Map.ply points and normals"],
            "does_not_use": [
                "restored Etoile mesh/object/material labels",
                "Sionna channel labels",
                "v212/v217 teacher outputs",
                "local replay truth",
                "hidden online labels",
            ],
        },
        "dependency": {"open3d": o3d.__version__, "purpose": "Poisson mesh reconstruction"},
        "config": vars(args),
        "point_cloud": {
            "original_count": int(original_count),
            "downsampled_count": int(downsampled_count),
        },
        "poisson_seconds": float(poisson_seconds),
        "density_threshold": threshold,
        "before_simplify": before_simplify,
        "after_simplify": after_simplify,
        "total_seconds": float(time.time() - started),
        "outputs": {"mesh": PATHS.relative(MESH), "report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
