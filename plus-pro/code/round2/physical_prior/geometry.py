"""Official-map loading and stable coplanar surface-family construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from project_paths import PATHS

from .config import PhysicalPriorConfig


MAP_PATH = PATHS.round2 / "Round2_Map.ply"
TRAIN_POS_PATH = PATHS.round2 / "Round2_Train_Pos.npy"
TEST_POS_PATH = PATHS.round2 / "Round2_Test_Pos.npy"


@dataclass(frozen=True)
class GeometryData:
    vertices: np.ndarray
    faces: np.ndarray
    face_family: np.ndarray
    family_normal: np.ndarray
    family_offset: np.ndarray
    family_origin: np.ndarray
    family_u: np.ndarray
    family_v: np.ndarray
    family_uv_min: np.ndarray
    family_uv_max: np.ndarray
    family_area: np.ndarray
    family_centroid: np.ndarray
    ground_family: int
    map_hash: str

    @property
    def family_count(self) -> int:
        return int(len(self.family_normal))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _ground_geometry(config: PhysicalPriorConfig) -> tuple[np.ndarray, np.ndarray]:
    train = np.load(TRAIN_POS_PATH, mmap_mode="r")
    test = np.load(TEST_POS_PATH, mmap_mode="r")
    positions = np.concatenate((train[:, :2], test[:, :2]), axis=0)
    low = positions.min(axis=0) - config.ground_padding_m
    high = positions.max(axis=0) + config.ground_padding_m
    vertices = np.asarray(
        (
            (low[0], low[1], 0.0),
            (high[0], low[1], 0.0),
            (high[0], high[1], 0.0),
            (low[0], high[1], 0.0),
        ),
        dtype=np.float32,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    return vertices, faces


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(normal, axis))) > 0.9:
        axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    u = np.cross(normal, axis)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(normal, u)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return u, v


def _surface_families(mesh: trimesh.Trimesh) -> tuple[np.ndarray, list[np.ndarray]]:
    face_family = np.full(len(mesh.faces), -1, dtype=np.int32)
    families: list[np.ndarray] = []
    for facet in mesh.facets:
        ids = np.sort(np.asarray(facet, dtype=np.int32))
        family = len(families)
        face_family[ids] = family
        families.append(ids)
    for face in np.flatnonzero(face_family < 0):
        family = len(families)
        face_family[face] = family
        families.append(np.asarray((face,), dtype=np.int32))
    return face_family, families


def build_geometry(config: PhysicalPriorConfig) -> GeometryData:
    if not MAP_PATH.is_file():
        raise FileNotFoundError(MAP_PATH)
    source = trimesh.load(MAP_PATH, process=False)
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(source)!r}")
    vertices = np.asarray(source.vertices, dtype=np.float32)
    faces = np.asarray(source.faces, dtype=np.int32)
    ground_vertices, ground_faces = _ground_geometry(config)
    ground_offset = len(vertices)
    vertices = np.concatenate((vertices, ground_vertices), axis=0)
    faces = np.concatenate((faces, ground_faces + ground_offset), axis=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    face_family, families = _surface_families(mesh)
    ground_family = int(face_family[-1])

    count = len(families)
    family_normal = np.zeros((count, 3), dtype=np.float32)
    family_offset = np.zeros(count, dtype=np.float32)
    family_origin = np.zeros((count, 3), dtype=np.float32)
    family_u = np.zeros((count, 3), dtype=np.float32)
    family_v = np.zeros((count, 3), dtype=np.float32)
    family_uv_min = np.zeros((count, 2), dtype=np.float32)
    family_uv_max = np.zeros((count, 2), dtype=np.float32)
    family_area = np.zeros(count, dtype=np.float32)
    family_centroid = np.zeros((count, 3), dtype=np.float32)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_area = np.asarray(mesh.area_faces, dtype=np.float64)
    for family, face_ids in enumerate(families):
        weights = np.maximum(face_area[face_ids], 1e-12)
        normal = np.average(face_normals[face_ids], axis=0, weights=weights)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        family_vertices = np.unique(faces[face_ids].reshape(-1))
        points = vertices[family_vertices].astype(np.float64)
        centroid = np.average(
            np.asarray(mesh.triangles_center)[face_ids], axis=0, weights=weights
        )
        u, v = _basis(normal)
        uv = np.stack(((points - centroid) @ u, (points - centroid) @ v), axis=1)
        family_normal[family] = normal.astype(np.float32)
        family_offset[family] = np.float32(-np.dot(normal, centroid))
        family_origin[family] = centroid.astype(np.float32)
        family_u[family] = u.astype(np.float32)
        family_v[family] = v.astype(np.float32)
        family_uv_min[family] = uv.min(axis=0).astype(np.float32)
        family_uv_max[family] = uv.max(axis=0).astype(np.float32)
        family_area[family] = np.float32(np.sum(weights))
        family_centroid[family] = centroid.astype(np.float32)

    return GeometryData(
        vertices=np.ascontiguousarray(vertices),
        faces=np.ascontiguousarray(faces),
        face_family=np.ascontiguousarray(face_family),
        family_normal=family_normal,
        family_offset=family_offset,
        family_origin=family_origin,
        family_u=family_u,
        family_v=family_v,
        family_uv_min=family_uv_min,
        family_uv_max=family_uv_max,
        family_area=family_area,
        family_centroid=family_centroid,
        ground_family=ground_family,
        map_hash=sha256(MAP_PATH),
    )


def save_geometry(path: Path, geometry: GeometryData, config: PhysicalPriorConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            vertices=geometry.vertices,
            faces=geometry.faces,
            face_family=geometry.face_family,
            family_normal=geometry.family_normal,
            family_offset=geometry.family_offset,
            family_origin=geometry.family_origin,
            family_u=geometry.family_u,
            family_v=geometry.family_v,
            family_uv_min=geometry.family_uv_min,
            family_uv_max=geometry.family_uv_max,
            family_area=geometry.family_area,
            family_centroid=geometry.family_centroid,
            ground_family=np.asarray(geometry.ground_family, dtype=np.int32),
            map_hash=np.asarray(geometry.map_hash),
            config_json=np.asarray(json.dumps(config.payload(), sort_keys=True)),
        )
    temporary.replace(path)


def load_geometry(path: Path) -> GeometryData:
    cache = np.load(path, allow_pickle=False)
    return GeometryData(
        vertices=cache["vertices"].astype(np.float32),
        faces=cache["faces"].astype(np.int32),
        face_family=cache["face_family"].astype(np.int32),
        family_normal=cache["family_normal"].astype(np.float32),
        family_offset=cache["family_offset"].astype(np.float32),
        family_origin=cache["family_origin"].astype(np.float32),
        family_u=cache["family_u"].astype(np.float32),
        family_v=cache["family_v"].astype(np.float32),
        family_uv_min=cache["family_uv_min"].astype(np.float32),
        family_uv_max=cache["family_uv_max"].astype(np.float32),
        family_area=cache["family_area"].astype(np.float32),
        family_centroid=cache["family_centroid"].astype(np.float32),
        ground_family=int(cache["ground_family"]),
        map_hash=str(cache["map_hash"]),
    )


def geometry_summary(geometry: GeometryData) -> dict[str, Any]:
    return {
        "vertices": int(len(geometry.vertices)),
        "triangles": int(len(geometry.faces)),
        "surface_families": int(geometry.family_count),
        "ground_family": int(geometry.ground_family),
        "map_sha256": geometry.map_hash,
        "family_area_quantiles": np.quantile(
            geometry.family_area.astype(np.float64), (0.0, 0.5, 0.9, 0.99, 1.0)
        ).tolist(),
    }
