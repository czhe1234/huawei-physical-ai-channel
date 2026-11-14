"""Exact triangle and quantized-plane data for source-aligned path solving."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .geometry import GeometryData


@dataclass(frozen=True)
class SourceAlignedGeometry:
    face_normal: np.ndarray
    face_offset: np.ndarray
    face_plane: np.ndarray
    plane_normal: np.ndarray
    plane_offset: np.ndarray
    digest: str

    @property
    def plane_count(self) -> int:
        return int(len(self.plane_offset))


def _canonical_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64).copy()
    flip_epsilon = 1e-4
    flip = (
        (normals[:, 2] < -flip_epsilon)
        | (
            (np.abs(normals[:, 2]) < flip_epsilon)
            & (normals[:, 1] < -flip_epsilon)
        )
        | (
            (np.abs(normals[:, 2]) < flip_epsilon)
            & (np.abs(normals[:, 1]) < flip_epsilon)
            & (normals[:, 0] < -flip_epsilon)
        )
    )
    normals[flip] *= -1.0
    return normals


def _quantize(values: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.abs(values) < epsilon, 0.0, values)
    return np.rint((values + 0.04281 * epsilon) / epsilon).astype(np.int64)


def build_source_aligned_geometry(
    geometry: GeometryData,
) -> SourceAlignedGeometry:
    triangles = geometry.vertices[geometry.faces].astype(np.float64)
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edges_a, edges_b)
    norm = np.linalg.norm(normals, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("source geometry contains degenerate triangles")
    normals /= norm[:, None]
    canonical = _canonical_normals(normals)
    offsets = np.einsum("fi,fi->f", canonical, triangles[:, 0])
    keys = np.concatenate(
        (
            _quantize(canonical, 1e-5),
            _quantize(offsets[:, None], 1e-3),
        ),
        axis=1,
    )
    _, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    plane_normal = canonical[first]
    plane_offset = offsets[first]
    digest = hashlib.sha256(
        np.ascontiguousarray(keys).view(np.uint8)
    ).hexdigest()
    return SourceAlignedGeometry(
        face_normal=np.ascontiguousarray(normals, dtype=np.float32),
        face_offset=np.ascontiguousarray(
            np.einsum("fi,fi->f", normals, triangles[:, 0]), dtype=np.float32
        ),
        face_plane=np.ascontiguousarray(inverse, dtype=np.int32),
        plane_normal=np.ascontiguousarray(plane_normal, dtype=np.float32),
        plane_offset=np.ascontiguousarray(plane_offset, dtype=np.float32),
        digest=digest,
    )


def save_source_aligned_geometry(
    path: Path, geometry: SourceAlignedGeometry
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            face_normal=geometry.face_normal,
            face_offset=geometry.face_offset,
            face_plane=geometry.face_plane,
            plane_normal=geometry.plane_normal,
            plane_offset=geometry.plane_offset,
            digest=np.asarray(geometry.digest),
        )
    temporary.replace(path)


def load_source_aligned_geometry(path: Path) -> SourceAlignedGeometry:
    with np.load(path, allow_pickle=False) as cache:
        return SourceAlignedGeometry(
            face_normal=cache["face_normal"].astype(np.float32),
            face_offset=cache["face_offset"].astype(np.float32),
            face_plane=cache["face_plane"].astype(np.int32),
            plane_normal=cache["plane_normal"].astype(np.float32),
            plane_offset=cache["plane_offset"].astype(np.float32),
            digest=str(cache["digest"].item()),
        )

