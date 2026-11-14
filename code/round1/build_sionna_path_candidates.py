#!/usr/bin/env python3
"""Build exact LOS and one-bounce specular candidates from Sionna Etoile meshes.

This is a geometry cache for learned models. It does not synthesize channels and
does not run Sionna ray tracing. It uses the known continuous mesh scene to
derive physically meaningful candidate descriptors:

- exact BS->RX segment intersections with scene triangles;
- exact image-source one-bounce specular reflections on mesh triangles;
- mesh/material/object identity for each selected candidate.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "trimesh is required. Run code/tools/bootstrap_environment.sh first."
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_map_features import BS_POS
from build_sionna_etoile_scene import (
    DEFAULT_SCENE_XML,
    MATERIAL_COLORS,
    MATERIAL_ORDER,
    ensure_dir,
    parse_scene_xml,
    percentile_dict,
)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-30)


def load_triangle_scene(scene_xml: Path) -> dict:
    shapes = parse_scene_xml(scene_xml)
    mat_to_id = {m: i for i, m in enumerate(MATERIAL_ORDER)}

    tri = []
    normal = []
    area = []
    mesh_id = []
    material_id = []
    mesh_names = []
    mesh_files = []
    mesh_material = []
    mesh_object = []
    mesh_bbox_min = []
    mesh_bbox_max = []
    mesh_vertex_count = []
    mesh_face_count = []

    for mid, shape in enumerate(shapes):
        path = Path(shape["path"])
        mesh = trimesh.load_mesh(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh for {path}, got {type(mesh)}")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.triangles, dtype=np.float64)
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        mat_id = mat_to_id.get(shape["material"], mat_to_id["unknown"])

        mesh_names.append(shape["shape_id"])
        mesh_files.append(shape["filename"])
        mesh_material.append(shape["material"])
        mesh_object.append(shape["object_name"])
        mesh_bbox_min.append(vertices.min(axis=0))
        mesh_bbox_max.append(vertices.max(axis=0))
        mesh_vertex_count.append(len(vertices))
        mesh_face_count.append(len(triangles))

        if len(triangles):
            tri.append(triangles.astype(np.float32))
            normal.append(normals.astype(np.float32))
            area.append(areas.astype(np.float32))
            mesh_id.append(np.full(len(triangles), mid, dtype=np.int32))
            material_id.append(np.full(len(triangles), mat_id, dtype=np.int16))

    triangles = np.concatenate(tri, axis=0)
    normals = np.concatenate(normal, axis=0)
    areas = np.concatenate(area, axis=0)
    mesh_ids = np.concatenate(mesh_id, axis=0)
    material_ids = np.concatenate(material_id, axis=0)
    centers = triangles.mean(axis=1)
    vertical = (centers[:, 2] > 1.0) & (np.abs(normals[:, 2]) < 0.7)
    non_ground = centers[:, 2] > 0.05

    return {
        "triangles": triangles,
        "tri_normals": normals,
        "tri_areas": areas,
        "tri_centers": centers.astype(np.float32),
        "tri_mesh_id": mesh_ids,
        "tri_material_id": material_ids,
        "tri_vertical_mask": vertical,
        "tri_non_ground_mask": non_ground,
        "mesh_names": np.asarray(mesh_names),
        "mesh_files": np.asarray(mesh_files),
        "mesh_material": np.asarray(mesh_material),
        "mesh_object": np.asarray(mesh_object),
        "mesh_bbox_min": np.asarray(mesh_bbox_min, dtype=np.float32),
        "mesh_bbox_max": np.asarray(mesh_bbox_max, dtype=np.float32),
        "mesh_vertex_count": np.asarray(mesh_vertex_count, dtype=np.int32),
        "mesh_face_count": np.asarray(mesh_face_count, dtype=np.int32),
        "material_names": np.asarray(MATERIAL_ORDER),
    }


def segment_triangle_hits(
    origins: np.ndarray,
    ends: np.ndarray,
    triangles: np.ndarray,
    tri_ids: np.ndarray,
    batch_triangles: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hit count, first-hit segment t, first-hit global triangle id."""

    origins = origins.astype(np.float32, copy=False)
    ends = ends.astype(np.float32, copy=False)
    d = ends - origins
    n_seg = len(origins)
    hit_count = np.zeros(n_seg, dtype=np.int32)
    first_t = np.full(n_seg, np.inf, dtype=np.float32)
    first_tri = np.full(n_seg, -1, dtype=np.int32)

    for start in range(0, len(triangles), batch_triangles):
        end = min(start + batch_triangles, len(triangles))
        tri = triangles[start:end].astype(np.float32, copy=False)
        ids = tri_ids[start:end]
        v0 = tri[:, 0, :]
        e1 = tri[:, 1, :] - tri[:, 0, :]
        e2 = tri[:, 2, :] - tri[:, 0, :]

        pvec = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum("ti,qti->qt", e1, pvec)
        det_ok = np.abs(det) > eps
        inv_det = np.where(det_ok, 1.0 / np.where(det_ok, det, 1.0), 0.0)

        tvec = origins[:, None, :] - v0[None, :, :]
        u = np.einsum("qti,qti->qt", tvec, pvec) * inv_det
        qvec = np.cross(tvec, e1[None, :, :])
        v = np.einsum("qi,qti->qt", d, qvec) * inv_det
        t = np.einsum("ti,qti->qt", e2, qvec) * inv_det

        hit = det_ok & (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps) & (t > eps) & (t < 1.0 - eps)
        hit_count += np.sum(hit, axis=1).astype(np.int32)
        local_t = np.where(hit, t, np.inf)
        best_local_idx = np.argmin(local_t, axis=1)
        best_local_t = local_t[np.arange(n_seg), best_local_idx]
        improve = best_local_t < first_t
        if np.any(improve):
            first_t[improve] = best_local_t[improve].astype(np.float32)
            first_tri[improve] = ids[best_local_idx[improve]].astype(np.int32)

    first_t[~np.isfinite(first_t)] = np.nan
    return hit_count, first_t, first_tri


def build_los(pos: np.ndarray, scene: dict, batch: int, batch_triangles: int) -> dict:
    triangles = scene["triangles"][scene["tri_non_ground_mask"]]
    tri_ids = np.nonzero(scene["tri_non_ground_mask"])[0].astype(np.int32)
    origins = np.repeat(BS_POS[None, :], len(pos), axis=0)
    counts = np.empty(len(pos), dtype=np.int32)
    first_t = np.empty(len(pos), dtype=np.float32)
    first_tri = np.empty(len(pos), dtype=np.int32)
    for start in range(0, len(pos), batch):
        end = min(start + batch, len(pos))
        c, t, tri = segment_triangle_hits(
            origins[start:end],
            pos[start:end],
            triangles,
            tri_ids,
            batch_triangles,
        )
        counts[start:end] = c
        first_t[start:end] = t
        first_tri[start:end] = tri
        print(f"processed exact LOS {end}/{len(pos)}", flush=True)
    first_mesh = np.full(len(pos), -1, dtype=np.int32)
    first_material = np.full(len(pos), -1, dtype=np.int16)
    first_point = np.full((len(pos), 3), np.nan, dtype=np.float32)
    valid = first_tri >= 0
    if np.any(valid):
        first_mesh[valid] = scene["tri_mesh_id"][first_tri[valid]]
        first_material[valid] = scene["tri_material_id"][first_tri[valid]]
        first_point[valid] = (
            BS_POS[None, :] * (1.0 - first_t[valid, None]) + pos[valid] * first_t[valid, None]
        ).astype(np.float32)
    return {
        "hit_count": counts,
        "clear": counts == 0,
        "first_t": first_t,
        "first_triangle_id": first_tri,
        "first_mesh_id": first_mesh,
        "first_material_id": first_material,
        "first_point": first_point,
    }


def barycentric_for_points(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]
    v0 = b - a
    v1 = c - a
    v2 = points - a[None, :, :]
    d00 = np.einsum("ti,ti->t", v0, v0)
    d01 = np.einsum("ti,ti->t", v0, v1)
    d11 = np.einsum("ti,ti->t", v1, v1)
    d20 = np.einsum("qti,ti->qt", v2, v0)
    d21 = np.einsum("qti,ti->qt", v2, v1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    v = (d11[None, :] * d20 - d01[None, :] * d21) / denom[None, :]
    w = (d00[None, :] * d21 - d01[None, :] * d20) / denom[None, :]
    u = 1.0 - v - w
    return u, v, w


def material_score(material_ids: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(MATERIAL_ORDER), dtype=np.float32)
    for i, name in enumerate(MATERIAL_ORDER):
        if name == "metal":
            scores[i] = 0.14
        elif name == "marble":
            scores[i] = 0.09
        elif name == "concrete":
            scores[i] = 0.06
        elif name == "wood":
            scores[i] = 0.02
    return scores[material_ids]


def build_specular(
    pos: np.ndarray,
    scene: dict,
    top_reflections: int,
    delay_slope: float,
    delay_intercept: float,
    batch: int,
) -> dict:
    keep = scene["tri_vertical_mask"]
    tri_ids = np.nonzero(keep)[0].astype(np.int32)
    triangles = scene["triangles"][keep].astype(np.float32, copy=False)
    normals = scene["tri_normals"][keep].astype(np.float32, copy=False)
    areas = scene["tri_areas"][keep].astype(np.float32, copy=False)
    mesh_id_all = scene["tri_mesh_id"][keep]
    material_id_all = scene["tri_material_id"][keep]
    n_tri = len(triangles)
    top = int(top_reflections)

    a = triangles[:, 0, :]
    normal = normals
    signed = np.einsum("ti,i->t", normal, BS_POS.astype(np.float32)) - np.einsum("ti,ti->t", normal, a)
    virtual_source_all = BS_POS.astype(np.float32)[None, :] - 2.0 * signed[:, None] * normal
    numerator = np.einsum("ti,ti->t", normal, a - virtual_source_all)
    area_score = np.log1p(areas) / max(float(np.log1p(areas).max()), 1e-12)
    mat_score = material_score(material_id_all)

    shape = (len(pos), top)
    valid = np.zeros(shape, dtype=np.bool_)
    triangle_id = np.full(shape, -1, dtype=np.int32)
    mesh_id = np.full(shape, -1, dtype=np.int32)
    material_id = np.full(shape, -1, dtype=np.int16)
    path_length = np.full(shape, np.inf, dtype=np.float32)
    delay_bin = np.full(shape, np.nan, dtype=np.float32)
    score = np.full(shape, -np.inf, dtype=np.float32)
    virtual_source = np.full((len(pos), top, 3), np.nan, dtype=np.float32)
    reflect_point = np.full((len(pos), top, 3), np.nan, dtype=np.float32)
    tx_dir = np.full((len(pos), top, 3), np.nan, dtype=np.float32)
    rx_dir = np.full((len(pos), top, 3), np.nan, dtype=np.float32)
    excess_length = np.full(shape, np.nan, dtype=np.float32)
    edge_margin = np.full(shape, np.nan, dtype=np.float32)
    incidence_cos = np.full(shape, np.nan, dtype=np.float32)

    for start in range(0, len(pos), batch):
        end = min(start + batch, len(pos))
        rx = pos[start:end].astype(np.float32, copy=False)
        ray = rx[:, None, :] - virtual_source_all[None, :, :]
        denom = np.einsum("qti,ti->qt", ray, normal)
        tau = numerator[None, :] / np.where(np.abs(denom) < 1e-8, np.nan, denom)
        point = virtual_source_all[None, :, :] + tau[:, :, None] * ray
        u, v, w = barycentric_for_points(point, triangles)
        edge = np.minimum(np.minimum(u, v), w)
        inside = (
            np.isfinite(tau)
            & (tau > 1e-5)
            & (tau < 1.0 - 1e-5)
            & (u >= -1e-5)
            & (v >= -1e-5)
            & (w >= -1e-5)
        )

        bs_to_ref = point - BS_POS.astype(np.float32)[None, None, :]
        ref_to_rx = rx[:, None, :] - point
        len1 = np.linalg.norm(bs_to_ref, axis=2)
        len2 = np.linalg.norm(ref_to_rx, axis=2)
        total_len = len1 + len2
        los_len = np.linalg.norm(rx - BS_POS.astype(np.float32)[None, :], axis=1)[:, None]
        excess = total_len - los_len
        inc = np.abs(np.einsum("qti,ti->qt", unit(-bs_to_ref), normal))
        edge_score = np.clip(3.0 * edge, -1.0, 1.0)

        raw_score = (
            3.0 * inside.astype(np.float32)
            + 0.22 * area_score[None, :]
            + mat_score[None, :]
            + 0.18 * edge_score
            + 0.08 * inc
            - 0.008 * np.maximum(excess, 0.0)
        )
        raw_score = np.where(inside, raw_score, -np.inf)
        take = min(top, n_tri)
        top_idx = np.argpartition(raw_score, -take, axis=1)[:, -take:]
        row = np.arange(end - start)[:, None]
        top_idx = top_idx[row, np.argsort(raw_score[row, top_idx], axis=1)[:, ::-1]]

        rows = np.arange(end - start)
        for j in range(take):
            idx = top_idx[:, j]
            sc = raw_score[rows, idx]
            good = np.isfinite(sc)
            valid[start:end, j] = good
            triangle_id[start:end, j] = tri_ids[idx]
            mesh_id[start:end, j] = mesh_id_all[idx]
            material_id[start:end, j] = material_id_all[idx]
            path_length[start:end, j] = total_len[rows, idx].astype(np.float32)
            delay_bin[start:end, j] = (delay_slope * total_len[rows, idx] + delay_intercept).astype(np.float32)
            score[start:end, j] = sc.astype(np.float32)
            virtual_source[start:end, j] = virtual_source_all[idx].astype(np.float32)
            reflect_point[start:end, j] = point[rows, idx].astype(np.float32)
            tx_dir[start:end, j] = unit(point[rows, idx] - BS_POS.astype(np.float32)[None, :]).astype(np.float32)
            rx_dir[start:end, j] = unit(point[rows, idx] - rx).astype(np.float32)
            excess_length[start:end, j] = excess[rows, idx].astype(np.float32)
            edge_margin[start:end, j] = edge[rows, idx].astype(np.float32)
            incidence_cos[start:end, j] = inc[rows, idx].astype(np.float32)

        invalid = ~valid[start:end]
        path_length[start:end][invalid] = np.inf
        delay_bin[start:end][invalid] = np.nan
        score[start:end][invalid] = -np.inf
        virtual_source[start:end][invalid] = np.nan
        reflect_point[start:end][invalid] = np.nan
        tx_dir[start:end][invalid] = np.nan
        rx_dir[start:end][invalid] = np.nan
        excess_length[start:end][invalid] = np.nan
        edge_margin[start:end][invalid] = np.nan
        incidence_cos[start:end][invalid] = np.nan

        print(f"processed exact specular {end}/{len(pos)}", flush=True)

    return {
        "valid": valid,
        "path_type": np.full(shape, 1, dtype=np.int8),
        "triangle_id": triangle_id,
        "mesh_id": mesh_id,
        "material_id": material_id,
        "path_length": path_length,
        "delay_bin": delay_bin,
        "score": score,
        "virtual_source": virtual_source,
        "reflect_point": reflect_point,
        "tx_dir": tx_dir,
        "rx_dir": rx_dir,
        "excess_length": excess_length,
        "edge_margin": edge_margin,
        "incidence_cos": incidence_cos,
    }


def combine_los_specular(
    pos: np.ndarray,
    los: dict,
    spec: dict,
    delay_slope: float,
    delay_intercept: float,
) -> dict:
    n = len(pos)
    top = spec["valid"].shape[1]
    shape = (n, 1 + top)
    valid = np.ones(shape, dtype=np.bool_)
    valid[:, 1:] = spec["valid"]
    path_type = np.zeros(shape, dtype=np.int8)
    path_type[:, 1:] = spec["path_type"]
    triangle_id = np.full(shape, -1, dtype=np.int32)
    mesh_id = np.full(shape, -1, dtype=np.int32)
    material_id = np.full(shape, -1, dtype=np.int16)
    path_length = np.full(shape, np.inf, dtype=np.float32)
    delay_bin = np.full(shape, np.nan, dtype=np.float32)
    score = np.full(shape, -np.inf, dtype=np.float32)
    virtual_source = np.full((n, 1 + top, 3), np.nan, dtype=np.float32)
    reflect_point = np.full((n, 1 + top, 3), np.nan, dtype=np.float32)
    tx_dir = np.full((n, 1 + top, 3), np.nan, dtype=np.float32)
    rx_dir = np.full((n, 1 + top, 3), np.nan, dtype=np.float32)
    excess_length = np.full(shape, np.nan, dtype=np.float32)
    edge_margin = np.full(shape, np.nan, dtype=np.float32)
    incidence_cos = np.full(shape, np.nan, dtype=np.float32)

    los_len = np.linalg.norm(pos - BS_POS[None, :], axis=1).astype(np.float32)
    path_length[:, 0] = los_len
    delay_bin[:, 0] = (delay_slope * los_len + delay_intercept).astype(np.float32)
    score[:, 0] = (1.0 / (1.0 + los["hit_count"].astype(np.float32))).astype(np.float32)
    virtual_source[:, 0] = BS_POS.astype(np.float32)
    tx_dir[:, 0] = unit(pos - BS_POS[None, :]).astype(np.float32)
    rx_dir[:, 0] = unit(BS_POS[None, :] - pos).astype(np.float32)
    excess_length[:, 0] = 0.0

    for key, arr in [
        ("triangle_id", triangle_id),
        ("mesh_id", mesh_id),
        ("material_id", material_id),
        ("path_length", path_length),
        ("delay_bin", delay_bin),
        ("score", score),
        ("virtual_source", virtual_source),
        ("reflect_point", reflect_point),
        ("tx_dir", tx_dir),
        ("rx_dir", rx_dir),
        ("excess_length", excess_length),
        ("edge_margin", edge_margin),
        ("incidence_cos", incidence_cos),
    ]:
        arr[:, 1:] = spec[key]

    return {
        "valid": valid,
        "path_type": path_type,
        "triangle_id": triangle_id,
        "mesh_id": mesh_id,
        "material_id": material_id,
        "path_length": path_length,
        "delay_bin": delay_bin,
        "score": score,
        "virtual_source": virtual_source,
        "reflect_point": reflect_point,
        "tx_dir": tx_dir,
        "rx_dir": rx_dir,
        "excess_length": excess_length,
        "edge_margin": edge_margin,
        "incidence_cos": incidence_cos,
        "los_hit_count": los["hit_count"],
        "los_clear": los["clear"],
        "los_first_t": los["first_t"],
        "los_first_triangle_id": los["first_triangle_id"],
        "los_first_mesh_id": los["first_mesh_id"],
        "los_first_material_id": los["first_material_id"],
        "los_first_point": los["first_point"],
    }


def summarize(name: str, cand: dict, material_names: np.ndarray, mesh_object: np.ndarray) -> dict:
    valid_ref = cand["valid"][:, 1:]
    top_valid = valid_ref[:, 0]
    summary = {
        "name": name,
        "points": int(cand["valid"].shape[0]),
        "candidate_count": int(cand["valid"].shape[1]),
        "los_clear_frac": float(np.mean(cand["los_clear"])),
        "los_hit_count": percentile_dict(cand["los_hit_count"]),
        "points_with_reflection": float(np.mean(np.any(valid_ref, axis=1))),
        "reflection_valid_frac": float(np.mean(valid_ref)),
        "top_reflection_valid_frac": float(np.mean(top_valid)),
    }
    if np.any(top_valid):
        summary.update(
            {
                "top_reflection_score": percentile_dict(np.where(top_valid, cand["score"][:, 1], np.nan)[top_valid]),
                "top_reflection_excess": percentile_dict(
                    np.where(top_valid, cand["excess_length"][:, 1], np.nan)[top_valid]
                ),
                "top_reflection_delay_bin": percentile_dict(
                    np.where(top_valid, cand["delay_bin"][:, 1], np.nan)[top_valid]
                ),
            }
        )
        mats = cand["material_id"][:, 1][top_valid]
        summary["top_reflection_material_counts"] = {
            str(material_names[int(k)]): int(v)
            for k, v in zip(*np.unique(mats, return_counts=True))
            if int(k) >= 0
        }
        meshes = cand["mesh_id"][:, 1][top_valid]
        mesh_counts = Counter(int(x) for x in meshes if int(x) >= 0)
        summary["top_reflection_objects"] = [
            {"mesh_id": int(mid), "object": str(mesh_object[mid]), "count": int(count)}
            for mid, count in mesh_counts.most_common(15)
        ]
    return summary


def plot_path_maps(fig_dir: Path, train_pos: np.ndarray, test_pos: np.ndarray, train: dict, test: dict, material_names) -> list[str]:
    out_files = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, pos, cand, title in [
        (axes[0], train_pos, train, "Train exact LOS hit count"),
        (axes[1], test_pos, test, "Test exact LOS hit count"),
    ]:
        sc = ax.scatter(pos[:, 0], pos[:, 1], s=12, c=cand["los_hit_count"], cmap="plasma", linewidths=0)
        ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=200, c="red", edgecolors="black", linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    out = fig_dir / "exact_los_hit_count.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    out_files.append(out.name)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, pos, cand, title in [
        (axes[0], train_pos, train, "Train top specular excess length"),
        (axes[1], test_pos, test, "Test top specular excess length"),
    ]:
        val = np.where(cand["valid"][:, 1], cand["excess_length"][:, 1], np.nan)
        sc = ax.scatter(pos[:, 0], pos[:, 1], s=12, c=val, cmap="viridis", linewidths=0)
        ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=200, c="red", edgecolors="black", linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    out = fig_dir / "top_reflection_excess.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    out_files.append(out.name)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    cmap = {i: MATERIAL_COLORS.get(str(name), "#9467bd") for i, name in enumerate(material_names)}
    for ax, pos, cand, title in [
        (axes[0], train_pos, train, "Train top specular material"),
        (axes[1], test_pos, test, "Test top specular material"),
    ]:
        mat = cand["material_id"][:, 1]
        colors = [cmap.get(int(x), "#dddddd") if cand["valid"][i, 1] else "#dddddd" for i, x in enumerate(mat)]
        ax.scatter(pos[:, 0], pos[:, 1], s=13, c=colors, linewidths=0)
        ax.scatter([BS_POS[0]], [BS_POS[1]], marker="*", s=200, c="red", edgecolors="black", linewidths=0.5)
        for i, name in enumerate(material_names):
            ax.scatter([], [], c=cmap.get(i, "#9467bd"), s=24, label=str(name))
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8)
    out = fig_dir / "top_reflection_material.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    out_files.append(out.name)
    return out_files


def write_analysis(out_dir: Path, summary: dict, figure_names: list[str]) -> None:
    lines = [
        "# Sionna Mesh Exact Path Candidates",
        "",
        "This cache uses the standard Sionna Etoile triangle meshes to build geometry descriptors for learned channel reconstruction. It is not a channel ray tracer; it only exports candidate path features.",
        "",
        "## Outputs",
        "",
        f"- Candidate cache: `{summary['output']}`",
        f"- Summary JSON: `{summary['summary']}`",
        "",
        "Figures:",
        "",
    ]
    for name in figure_names:
        lines.append(f"- `figures/{name}`")
    lines.extend(
        [
            "",
            "## Train/Test Summary",
            "",
            f"- Train LOS clear fraction: `{summary['train']['los_clear_frac']:.4f}`",
            f"- Test LOS clear fraction: `{summary['test']['los_clear_frac']:.4f}`",
            f"- Train points with specular candidate: `{summary['train']['points_with_reflection']:.4f}`",
            f"- Test points with specular candidate: `{summary['test']['points_with_reflection']:.4f}`",
            "",
            "## Interpretation",
            "",
            "Compared with the old `wall_planes.json` candidates, these candidates are tied to exact mesh triangles and retain material/object identity. The next modeling step is to use these fields as features for AD/component residual experts, especially for path-family gating and material-specific scalar correction.",
        ]
    )
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-xml", default=DEFAULT_SCENE_XML)
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--delay-fit", default=str(PATHS.artifacts / "shared_features/delay_length_fit.npz"))
    parser.add_argument("--output", default=str(PATHS.artifacts / "shared_features/sionna_path_candidates.npz"))
    parser.add_argument("--summary", default=str(PATHS.analysis / "results/json/sionna_path_candidates_summary.json"))
    parser.add_argument("--viz-dir", default=str(PATHS.analysis / "visualizations/sionna_path_candidates"))
    parser.add_argument("--top-reflections", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--los-batch", type=int, default=96)
    parser.add_argument("--batch-triangles", type=int, default=2048)
    args = parser.parse_args()

    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    delay = np.load(args.delay_fit)
    delay_slope = float(delay["slope"])
    delay_intercept = float(delay["intercept"])

    print(f"loading triangle scene: {args.scene_xml}", flush=True)
    scene = load_triangle_scene(Path(args.scene_xml))
    print(
        f"loaded triangles={len(scene['triangles'])} vertical={int(np.sum(scene['tri_vertical_mask']))}",
        flush=True,
    )

    payload = {
        "bs_pos": BS_POS.astype(np.float32),
        "delay_slope": np.asarray(delay_slope, dtype=np.float32),
        "delay_intercept": np.asarray(delay_intercept, dtype=np.float32),
        "mesh_names": scene["mesh_names"],
        "mesh_files": scene["mesh_files"],
        "mesh_material": scene["mesh_material"],
        "mesh_object": scene["mesh_object"],
        "material_names": scene["material_names"],
        "tri_centers": scene["tri_centers"],
        "tri_normals": scene["tri_normals"],
        "tri_areas": scene["tri_areas"],
        "tri_mesh_id": scene["tri_mesh_id"],
        "tri_material_id": scene["tri_material_id"],
        "tri_vertical_mask": scene["tri_vertical_mask"],
    }
    summaries = {}
    candidates = {}
    for prefix, pos in [("train", train_pos), ("test", test_pos)]:
        print(f"building {prefix} LOS", flush=True)
        los = build_los(pos, scene, args.los_batch, args.batch_triangles)
        print(f"building {prefix} exact specular", flush=True)
        spec = build_specular(pos, scene, args.top_reflections, delay_slope, delay_intercept, args.batch)
        cand = combine_los_specular(pos, los, spec, delay_slope, delay_intercept)
        candidates[prefix] = cand
        summaries[prefix] = summarize(prefix, cand, scene["material_names"], scene["mesh_object"])
        for key, value in cand.items():
            payload[f"{prefix}_{key}"] = value

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)

    summary = {
        "output": str(out),
        "summary": str(args.summary),
        "scene_xml": str(args.scene_xml),
        "delay_slope": delay_slope,
        "delay_intercept": delay_intercept,
        "top_reflections": int(args.top_reflections),
        "triangle_count": int(len(scene["triangles"])),
        "vertical_triangle_count": int(np.sum(scene["tri_vertical_mask"])),
        "material_counts": dict(Counter(str(x) for x in scene["mesh_material"])),
        "train": summaries["train"],
        "test": summaries["test"],
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))

    viz_dir = Path(args.viz_dir)
    fig_dir = viz_dir / "figures"
    ensure_dir(fig_dir)
    figures = plot_path_maps(fig_dir, train_pos, test_pos, candidates["train"], candidates["test"], scene["material_names"])
    write_analysis(viz_dir, summary, figures)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
