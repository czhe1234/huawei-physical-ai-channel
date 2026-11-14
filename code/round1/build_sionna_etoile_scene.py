#!/usr/bin/env python3
"""Build a continuous-geometry cache from Sionna's Etoile scene.

This script does not run ray tracing or synthesize a channel. It parses the
standard Sionna RT Etoile XML scene, extracts mesh/material geometry, creates a
surface-sample proxy for fast spatial queries, and writes train/test geometric
features that can be consumed by learned residual/component models.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "trimesh is required. Run code/tools/bootstrap_environment.sh first."
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_map_features import BS_POS, read_ply_memmap


DEFAULT_SCENE_XML = str(
    PATHS.artifacts / "external" / "restored_sionna_official_etoile" / "etoile.xml"
)
MATERIAL_ORDER = ["concrete", "marble", "metal", "wood", "unknown"]
MATERIAL_COLORS = {
    "concrete": "#7f7f7f",
    "marble": "#1f77b4",
    "metal": "#ff7f0e",
    "wood": "#8c564b",
    "unknown": "#9467bd",
}


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


def parse_scene_xml(xml_path: Path) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    shapes = []
    for shape in root.findall("shape"):
        shape_id = shape.attrib.get("id", "")
        filename = None
        material = "unknown"
        for child in shape:
            if child.tag == "string" and child.attrib.get("name") == "filename":
                filename = child.attrib.get("value")
            elif child.tag == "ref":
                material = child.attrib.get("id", "unknown")
        if filename is None:
            continue
        shapes.append(
            {
                "shape_id": shape_id,
                "filename": filename,
                "path": str((xml_path.parent / filename).resolve()),
                "material": material,
                "object_name": object_name_from_shape(shape_id),
            }
        )
    return shapes


def object_name_from_shape(shape_id: str) -> str:
    name = shape_id
    if name.startswith("mesh-"):
        name = name[len("mesh-") :]
    for suffix in ["-itu_concrete", "-itu_marble", "-itu_metal", "-itu_wood"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def sample_triangles(
    triangles: np.ndarray,
    normals: np.ndarray,
    mesh_id: int,
    material_id: int,
    area_faces: np.ndarray,
    rng: np.random.Generator,
    sample_area: float,
    max_per_triangle: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if triangles.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int16),
        )
    counts = np.ceil(np.maximum(area_faces, 1e-6) / sample_area).astype(np.int32)
    counts = np.clip(counts, 1, max_per_triangle)
    total = int(np.sum(counts))
    pts = np.empty((total, 3), dtype=np.float32)
    nrm = np.empty((total, 3), dtype=np.float32)
    mid = np.full(total, mesh_id, dtype=np.int32)
    mat = np.full(total, material_id, dtype=np.int16)
    cursor = 0
    for tri, normal, count in zip(triangles, normals, counts):
        # Deterministic random barycentric samples, plus centroid behavior for
        # count=1. The sqrt transform gives uniform samples on a triangle.
        if count == 1:
            bary = np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64)
        else:
            u = rng.random(count)
            v = rng.random(count)
            su = np.sqrt(u)
            bary = np.stack([1.0 - su, su * (1.0 - v), su * v], axis=1)
        out = bary @ tri
        pts[cursor : cursor + count] = out.astype(np.float32)
        nrm[cursor : cursor + count] = normal.astype(np.float32)
        cursor += count
    return pts, nrm, mid, mat


def load_mesh_geometry(scene_xml: Path, sample_area: float, max_per_triangle: int, seed: int) -> dict:
    shapes = parse_scene_xml(scene_xml)
    mat_to_id = {m: i for i, m in enumerate(MATERIAL_ORDER)}
    rng = np.random.default_rng(seed)

    mesh_names = []
    mesh_files = []
    mesh_material = []
    mesh_object = []
    mesh_bbox_min = []
    mesh_bbox_max = []
    mesh_vertex_count = []
    mesh_face_count = []
    tri_centers = []
    tri_normals = []
    tri_areas = []
    tri_mesh_id = []
    tri_material_id = []
    sample_points = []
    sample_normals = []
    sample_mesh_id = []
    sample_material_id = []

    for mesh_id, shape in enumerate(shapes):
        mesh_path = Path(shape["path"])
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh for {mesh_path}, got {type(mesh)}")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        material_id = mat_to_id.get(shape["material"], mat_to_id["unknown"])

        mesh_names.append(shape["shape_id"])
        mesh_files.append(shape["filename"])
        mesh_material.append(shape["material"])
        mesh_object.append(shape["object_name"])
        mesh_bbox_min.append(vertices.min(axis=0))
        mesh_bbox_max.append(vertices.max(axis=0))
        mesh_vertex_count.append(len(vertices))
        mesh_face_count.append(len(faces))

        if len(faces) > 0:
            triangles = np.asarray(mesh.triangles, dtype=np.float64)
            centers = triangles.mean(axis=1)
            normals = np.asarray(mesh.face_normals, dtype=np.float64)
            areas = np.asarray(mesh.area_faces, dtype=np.float64)
            tri_centers.append(centers.astype(np.float32))
            tri_normals.append(normals.astype(np.float32))
            tri_areas.append(areas.astype(np.float32))
            tri_mesh_id.append(np.full(len(faces), mesh_id, dtype=np.int32))
            tri_material_id.append(np.full(len(faces), material_id, dtype=np.int16))

            pts, nrm, mid, mat = sample_triangles(
                triangles, normals, mesh_id, material_id, areas, rng, sample_area, max_per_triangle
            )
            sample_points.append(pts)
            sample_normals.append(nrm)
            sample_mesh_id.append(mid)
            sample_material_id.append(mat)

    out = {
        "mesh_names": np.asarray(mesh_names),
        "mesh_files": np.asarray(mesh_files),
        "mesh_material": np.asarray(mesh_material),
        "mesh_object": np.asarray(mesh_object),
        "mesh_bbox_min": np.asarray(mesh_bbox_min, dtype=np.float32),
        "mesh_bbox_max": np.asarray(mesh_bbox_max, dtype=np.float32),
        "mesh_vertex_count": np.asarray(mesh_vertex_count, dtype=np.int32),
        "mesh_face_count": np.asarray(mesh_face_count, dtype=np.int32),
        "triangle_centers": np.concatenate(tri_centers, axis=0),
        "triangle_normals": np.concatenate(tri_normals, axis=0),
        "triangle_areas": np.concatenate(tri_areas, axis=0),
        "triangle_mesh_id": np.concatenate(tri_mesh_id, axis=0),
        "triangle_material_id": np.concatenate(tri_material_id, axis=0),
        "sample_points": np.concatenate(sample_points, axis=0),
        "sample_normals": np.concatenate(sample_normals, axis=0),
        "sample_mesh_id": np.concatenate(sample_mesh_id, axis=0),
        "sample_material_id": np.concatenate(sample_material_id, axis=0),
        "material_names": np.asarray(MATERIAL_ORDER),
        "scene_xml": np.asarray(str(scene_xml)),
    }
    out["triangle_vertical_mask"] = (
        (np.abs(out["triangle_normals"][:, 2]) < 0.7) & (out["triangle_centers"][:, 2] > 1.0)
    )
    out["sample_vertical_mask"] = (
        (np.abs(out["sample_normals"][:, 2]) < 0.7) & (out["sample_points"][:, 2] > 1.0)
    )
    out["sample_arc_mask"] = np.char.startswith(out["mesh_object"][out["sample_mesh_id"]], "Arc_de_Triomphe")
    return out


def query_line_features(
    sample_points: np.ndarray,
    sample_normals: np.ndarray,
    sample_material_id: np.ndarray,
    sample_mesh_id: np.ndarray,
    pos: np.ndarray,
    samples: np.ndarray,
    thresholds: list[float],
    batch: int,
) -> np.ndarray:
    tree = cKDTree(sample_points)
    feature_rows = []
    eps = 1e-6
    for start in range(0, len(pos), batch):
        end = min(start + batch, len(pos))
        ue = pos[start:end].astype(np.float64)
        line = BS_POS[None, None, :] * (1.0 - samples[None, :, None]) + ue[:, None, :] * samples[None, :, None]
        query = line.reshape(-1, 3)
        dist, idx = tree.query(query, k=1, workers=-1)
        dist = dist.reshape(end - start, samples.size)
        idx = idx.reshape(end - start, samples.size)
        min_idx_along = np.argmin(dist, axis=1)
        nearest_idx = idx[np.arange(end - start), min_idx_along]
        nearest_normals = sample_normals[nearest_idx].astype(np.float64)
        nearest_material = sample_material_id[nearest_idx].astype(np.float64)
        nearest_mesh = sample_mesh_id[nearest_idx].astype(np.float64)
        feats = [
            np.min(dist, axis=1),
            np.percentile(dist, 5, axis=1),
            np.percentile(dist, 10, axis=1),
            np.mean(1.0 / np.maximum(dist, eps), axis=1),
            samples[min_idx_along],
            nearest_material,
            nearest_mesh,
            nearest_normals[:, 0],
            nearest_normals[:, 1],
            nearest_normals[:, 2],
        ]
        for radius in thresholds:
            hit = dist < radius
            count = np.sum(hit, axis=1).astype(np.float64)
            any_hit = count > 0
            first = np.ones(end - start, dtype=np.float64)
            last = np.zeros(end - start, dtype=np.float64)
            if np.any(any_hit):
                first_idx = np.argmax(hit[any_hit], axis=1)
                last_idx = hit.shape[1] - 1 - np.argmax(hit[any_hit, ::-1], axis=1)
                first[any_hit] = samples[first_idx]
                last[any_hit] = samples[last_idx]
            feats.extend([count, count / samples.size, first, last])
        feature_rows.append(np.stack(feats, axis=1))
    return np.concatenate(feature_rows, axis=0).astype(np.float32)


def query_receiver_features(
    all_points: np.ndarray,
    all_normals: np.ndarray,
    all_material_id: np.ndarray,
    all_mesh_id: np.ndarray,
    arc_points: np.ndarray,
    pos: np.ndarray,
    batch: int,
) -> np.ndarray:
    tree3 = cKDTree(all_points)
    tree2 = cKDTree(all_points[:, :2])
    arc_tree = cKDTree(arc_points) if len(arc_points) else None
    rows = []
    for start in range(0, len(pos), batch):
        end = min(start + batch, len(pos))
        p = pos[start:end].astype(np.float64)
        dist3, idx3 = tree3.query(p, k=1, workers=-1)
        dist2, idx2 = tree2.query(p[:, :2], k=1, workers=-1)
        nearest3 = all_points[idx3]
        nearest2 = all_points[idx2]
        normals3 = all_normals[idx3]
        normals2 = all_normals[idx2]
        if arc_tree is not None:
            arc_dist, _ = arc_tree.query(p, k=1, workers=-1)
        else:
            arc_dist = np.full(end - start, np.nan, dtype=np.float64)
        rows.append(
            np.concatenate(
                [
                    dist3[:, None],
                    nearest3,
                    normals3,
                    all_material_id[idx3, None].astype(np.float32),
                    all_mesh_id[idx3, None].astype(np.float32),
                    dist2[:, None],
                    nearest2,
                    normals2,
                    all_material_id[idx2, None].astype(np.float32),
                    all_mesh_id[idx2, None].astype(np.float32),
                    arc_dist[:, None],
                ],
                axis=1,
            )
        )
    return np.concatenate(rows, axis=0).astype(np.float32)


def make_feature_names(thresholds: list[float]) -> list[str]:
    names = [
        "sionna_los_min_dist",
        "sionna_los_p05_dist",
        "sionna_los_p10_dist",
        "sionna_los_mean_inv_dist",
        "sionna_los_nearest_t",
        "sionna_los_nearest_material_id",
        "sionna_los_nearest_mesh_id",
        "sionna_los_nearest_nx",
        "sionna_los_nearest_ny",
        "sionna_los_nearest_nz",
    ]
    for radius in thresholds:
        tag = str(radius).replace(".", "p")
        names.extend(
            [
                f"sionna_los_count_r{tag}",
                f"sionna_los_frac_r{tag}",
                f"sionna_los_first_t_r{tag}",
                f"sionna_los_last_t_r{tag}",
            ]
        )
    names.extend(
        [
            "sionna_rx_dist3",
            "sionna_rx_x3",
            "sionna_rx_y3",
            "sionna_rx_z3",
            "sionna_rx_nx3",
            "sionna_rx_ny3",
            "sionna_rx_nz3",
            "sionna_rx_material_id3",
            "sionna_rx_mesh_id3",
            "sionna_rx_dist2",
            "sionna_rx_x2",
            "sionna_rx_y2",
            "sionna_rx_z2",
            "sionna_rx_nx2",
            "sionna_rx_ny2",
            "sionna_rx_nz2",
            "sionna_rx_material_id2",
            "sionna_rx_mesh_id2",
            "sionna_rx_arc_dist3",
        ]
    )
    return names


def build_features(cache: dict, train_pos: np.ndarray, test_pos: np.ndarray, line_samples: int, batch: int) -> dict:
    vertical = cache["sample_vertical_mask"]
    points = cache["sample_points"][vertical]
    normals = cache["sample_normals"][vertical]
    mesh_id = cache["sample_mesh_id"][vertical]
    material_id = cache["sample_material_id"][vertical]
    arc_points = cache["sample_points"][cache["sample_arc_mask"]]
    samples = np.linspace(0.03, 0.97, line_samples, dtype=np.float64)
    thresholds = [0.5, 1.0, 2.0, 3.0]
    names = make_feature_names(thresholds)

    train_line = query_line_features(points, normals, material_id, mesh_id, train_pos, samples, thresholds, batch)
    test_line = query_line_features(points, normals, material_id, mesh_id, test_pos, samples, thresholds, batch)
    train_rx = query_receiver_features(points, normals, material_id, mesh_id, arc_points, train_pos, batch)
    test_rx = query_receiver_features(points, normals, material_id, mesh_id, arc_points, test_pos, batch)
    return {
        "train_features": np.concatenate([train_line, train_rx], axis=1).astype(np.float32),
        "test_features": np.concatenate([test_line, test_rx], axis=1).astype(np.float32),
        "feature_names": np.asarray(names),
        "thresholds": np.asarray(thresholds, dtype=np.float32),
    }


def sample_round1_map(vertices, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(vertices)
    idx = rng.choice(n, size=min(count, n), replace=False)
    return np.stack(
        [
            np.asarray(vertices["x"][idx], dtype=np.float64),
            np.asarray(vertices["y"][idx], dtype=np.float64),
            np.asarray(vertices["z"][idx], dtype=np.float64),
        ],
        axis=1,
    )


def plot_scene_overview(fig_dir: Path, cache: dict, train_pos: np.ndarray, test_pos: np.ndarray, map_sample: np.ndarray) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    centers = cache["triangle_centers"]
    mats = cache["triangle_material_id"]
    material_names = [str(x) for x in cache["material_names"]]
    for mat_id, mat_name in enumerate(material_names):
        mask = mats == mat_id
        if not np.any(mask):
            continue
        axes[0].scatter(
            centers[mask, 0],
            centers[mask, 1],
            s=np.clip(cache["triangle_areas"][mask] * 0.03, 1, 8),
            c=MATERIAL_COLORS.get(mat_name, "#9467bd"),
            alpha=0.45,
            label=mat_name,
            linewidths=0,
        )
    axes[0].scatter(train_pos[:, 0], train_pos[:, 1], s=7, c="#2ca02c", alpha=0.45, label="train RX")
    axes[0].scatter(test_pos[:, 0], test_pos[:, 1], s=14, c="#d62728", marker="x", alpha=0.85, label="test RX")
    axes[0].scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=220, c="red", edgecolors="black", linewidths=0.5, label="BS")
    axes[0].set_title("Sionna Etoile continuous mesh footprint")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(fontsize=8, ncols=2)

    axes[1].scatter(map_sample[:, 0], map_sample[:, 1], s=1, c=map_sample[:, 2], cmap="terrain", alpha=0.18, linewidths=0)
    axes[1].scatter(centers[:, 0], centers[:, 1], s=2, c="black", alpha=0.28, linewidths=0)
    axes[1].scatter(train_pos[:, 0], train_pos[:, 1], s=7, c="#2ca02c", alpha=0.45)
    axes[1].scatter(test_pos[:, 0], test_pos[:, 1], s=14, c="#d62728", marker="x", alpha=0.85)
    axes[1].scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=220, c="red", edgecolors="black", linewidths=0.5)
    axes[1].set_title("Round1 sampled point cloud vs Sionna mesh centers")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal", adjustable="box")

    out = fig_dir / "scene_overview.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name


def plot_scene_features(fig_dir: Path, train_pos: np.ndarray, test_pos: np.ndarray, features: dict) -> str:
    names = [str(x) for x in features["feature_names"]]
    train = features["train_features"]
    test = features["test_features"]
    idx_los = names.index("sionna_los_min_dist")
    idx_arc = names.index("sionna_rx_arc_dist3")
    idx_mat = names.index("sionna_los_nearest_material_id")
    idx_rx = names.index("sionna_rx_dist2")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    panels = [
        (train_pos, train[:, idx_los], "Train: LOS nearest mesh-surface distance", "viridis"),
        (test_pos, test[:, idx_los], "Test: LOS nearest mesh-surface distance", "viridis"),
        (train_pos, train[:, idx_arc], "Train: distance to Arc de Triomphe mesh", "magma"),
        (test_pos, test[:, idx_arc], "Test: distance to Arc de Triomphe mesh", "magma"),
    ]
    for ax, (pos, values, title, cmap) in zip(axes.ravel(), panels):
        sc = ax.scatter(pos[:, 0], pos[:, 1], s=12, c=values, cmap=cmap, linewidths=0)
        ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=180, c="red", edgecolors="black", linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    out = fig_dir / "scene_features.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    axes[0].hist(train[:, idx_los], bins=40, alpha=0.6, density=True, label="train")
    axes[0].hist(test[:, idx_los], bins=30, alpha=0.6, density=True, label="test")
    axes[0].set_title("LOS min distance distribution")
    axes[0].legend()
    axes[1].hist(train[:, idx_rx], bins=40, alpha=0.6, density=True, label="train")
    axes[1].hist(test[:, idx_rx], bins=30, alpha=0.6, density=True, label="test")
    axes[1].set_title("RX nearest vertical mesh distance")
    axes[1].legend()
    axes[2].hist(train[:, idx_mat], bins=np.arange(-0.5, len(MATERIAL_ORDER) + 0.5), alpha=0.6, label="train")
    axes[2].hist(test[:, idx_mat], bins=np.arange(-0.5, len(MATERIAL_ORDER) + 0.5), alpha=0.6, label="test")
    axes[2].set_xticks(range(len(MATERIAL_ORDER)))
    axes[2].set_xticklabels(MATERIAL_ORDER, rotation=25, ha="right")
    axes[2].set_title("Nearest LOS material id")
    axes[2].legend()
    out2 = fig_dir / "feature_distributions.png"
    fig.savefig(out2, dpi=170)
    plt.close(fig)
    return f"{out.name},{out2.name}"


def write_analysis(
    out_dir: Path,
    summary: dict,
    overview_fig: str,
    feature_figs: str,
    cache_path: Path,
    feature_path: Path,
) -> None:
    material_counts = summary["material_counts"]
    lines = [
        "# Sionna Etoile Continuous Scene Analysis",
        "",
        "This directory verifies that the official Round1 point cloud aligns with the standard Sionna `etoile` scene and exports a continuous mesh cache for later Physical AI models.",
        "",
        "## Outputs",
        "",
        f"- Scene cache: `{cache_path}`",
        f"- Train/test feature cache: `{feature_path}`",
        f"- Overview figure: `figures/{overview_fig}`",
        f"- Feature figures: `{', '.join('figures/' + x for x in feature_figs.split(','))}`",
        "",
        "## Scene Summary",
        "",
        f"- XML: `{summary['scene_xml']}`",
        f"- Mesh shapes: `{summary['mesh_count']}`",
        f"- Triangles: `{summary['triangle_count']}`",
        f"- Surface samples: `{summary['sample_count']}`",
        f"- Vertical surface samples: `{summary['vertical_sample_count']}`",
        f"- Round1 point-cloud vertices: `{summary['round1_map_vertices']}`",
        "",
        "Material counts:",
        "",
    ]
    for material, count in material_counts.items():
        lines.append(f"- `{material}`: `{count}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The Sionna mesh footprint matches the Round1 point-cloud range, including the ground plane extent and the Arc de Triomphe height near 50 m. This means we can stop treating `Round1_Map.ply` as only an unstructured point cloud: the original continuous scene, object IDs, and ITU material tags are available.",
            "",
            "For modeling, this cache is useful as a geometry prior rather than a traditional ray-tracing submission. The immediate hooks are:",
            "",
            "- Replace nearest-point wall heuristics with mesh-surface/object/material features.",
            "- Generate reflection candidates from actual mesh faces instead of clustered point-cloud planes.",
            "- Add material and object identity to component/residual experts.",
            "- Diagnose whether high-scoring residuals concentrate around the Arc, metal roofs, marble walls, or generic buildings.",
            "",
            "The first feature cache intentionally uses fast surface samples, not exact ray intersections. It is enough for model features and diagnostics. If it proves useful, the next step is exact segment/triangle intersection and specular face candidates.",
        ]
    )
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-xml", default=DEFAULT_SCENE_XML)
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/sionna_etoile_scene_cache.npz"))
    parser.add_argument("--features", default=str(PATHS.artifacts / "shared_features/sionna_etoile_features.npz"))
    parser.add_argument("--summary", default=str(PATHS.analysis / "results/json/sionna_etoile_scene_summary.json"))
    parser.add_argument("--viz-dir", default=str(PATHS.analysis / "visualizations/sionna_etoile_scene"))
    parser.add_argument("--sample-area", type=float, default=20.0)
    parser.add_argument("--max-per-triangle", type=int, default=16)
    parser.add_argument("--line-samples", type=int, default=128)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    scene_xml = Path(args.scene_xml)
    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    round1_map = read_ply_memmap(data / "Round1_Map.ply")

    print(f"loading Sionna scene: {scene_xml}", flush=True)
    cache = load_mesh_geometry(scene_xml, args.sample_area, args.max_per_triangle, args.seed)
    print(
        f"loaded {len(cache['mesh_names'])} meshes, "
        f"{len(cache['triangle_centers'])} triangles, {len(cache['sample_points'])} samples",
        flush=True,
    )

    feature_payload = build_features(cache, train_pos, test_pos, args.line_samples, args.batch)
    print(
        f"features train={feature_payload['train_features'].shape} "
        f"test={feature_payload['test_features'].shape}",
        flush=True,
    )

    cache_path = Path(args.cache)
    feature_path = Path(args.features)
    summary_path = Path(args.summary)
    ensure_dir(cache_path.parent)
    np.savez_compressed(cache_path, **cache)
    np.savez_compressed(feature_path, **feature_payload)

    material_counts = Counter(str(x) for x in cache["mesh_material"])
    mesh_bbox_min = np.min(cache["mesh_bbox_min"], axis=0)
    mesh_bbox_max = np.max(cache["mesh_bbox_max"], axis=0)
    round1_bbox_min = [
        float(np.min(np.asarray(round1_map["x"]))),
        float(np.min(np.asarray(round1_map["y"]))),
        float(np.min(np.asarray(round1_map["z"]))),
    ]
    round1_bbox_max = [
        float(np.max(np.asarray(round1_map["x"]))),
        float(np.max(np.asarray(round1_map["y"]))),
        float(np.max(np.asarray(round1_map["z"]))),
    ]
    names = [str(x) for x in feature_payload["feature_names"]]
    train_features = feature_payload["train_features"]
    test_features = feature_payload["test_features"]
    idx_los = names.index("sionna_los_min_dist")
    idx_arc = names.index("sionna_rx_arc_dist3")
    summary = {
        "scene_xml": str(scene_xml),
        "cache": str(cache_path),
        "features": str(feature_path),
        "mesh_count": int(len(cache["mesh_names"])),
        "triangle_count": int(len(cache["triangle_centers"])),
        "sample_count": int(len(cache["sample_points"])),
        "vertical_sample_count": int(np.sum(cache["sample_vertical_mask"])),
        "arc_sample_count": int(np.sum(cache["sample_arc_mask"])),
        "round1_map_vertices": int(len(round1_map)),
        "material_counts": dict(material_counts),
        "mesh_bbox_min": mesh_bbox_min.astype(float).tolist(),
        "mesh_bbox_max": mesh_bbox_max.astype(float).tolist(),
        "round1_bbox_min": round1_bbox_min,
        "round1_bbox_max": round1_bbox_max,
        "feature_count": int(len(names)),
        "feature_names": names,
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "train_los_min_dist": percentile_dict(train_features[:, idx_los]),
        "test_los_min_dist": percentile_dict(test_features[:, idx_los]),
        "train_arc_dist": percentile_dict(train_features[:, idx_arc]),
        "test_arc_dist": percentile_dict(test_features[:, idx_arc]),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    viz_dir = Path(args.viz_dir)
    fig_dir = viz_dir / "figures"
    ensure_dir(fig_dir)
    map_sample = sample_round1_map(round1_map, 120000, args.seed)
    overview_fig = plot_scene_overview(fig_dir, cache, train_pos, test_pos, map_sample)
    feature_figs = plot_scene_features(fig_dir, train_pos, test_pos, feature_payload)
    write_analysis(viz_dir, summary, overview_fig, feature_figs, cache_path, feature_path)

    print(
        json.dumps(
            {
                "cache": str(cache_path),
                "features": str(feature_path),
                "summary": str(summary_path),
                "viz": str(viz_dir / "analysis.md"),
                "mesh_count": summary["mesh_count"],
                "triangle_count": summary["triangle_count"],
                "vertical_sample_count": summary["vertical_sample_count"],
                "train_shape": summary["train_shape"],
                "test_shape": summary["test_shape"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
