"""Shared production configuration and compact path-support extraction."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
from scipy.constants import speed_of_light

from project_paths import PATHS


E185_SEEDS = (42, 101, 211)
SCENE_DIR = PATHS.artifacts / "round2" / "e209_source_port" / "scene"


@dataclass(frozen=True)
class PropagationConfig:
    name: str
    frequency_hz: float
    site_yaw_deg: tuple[float, float]
    site_tilt_deg: tuple[float, float]
    tx_vertical_spacing: float
    tx_horizontal_spacing: float
    max_depth: int
    synthetic_array: bool
    diffuse_reflection: bool
    scattering_coefficient: float
    diffraction: bool
    edge_diffraction: bool
    samples_per_src: int
    max_num_paths_per_src: int
    source_code: int


CONFIGS = {
    "base": PropagationConfig(
        name="primary_d16_base_v4lambda_1m",
        frequency_hz=6.69e9,
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
        tx_vertical_spacing=4.0,
        tx_horizontal_spacing=0.5,
        max_depth=16,
        synthetic_array=True,
        diffuse_reflection=False,
        scattering_coefficient=0.0,
        diffraction=False,
        edge_diffraction=False,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        source_code=0,
    ),
    "scatter": PropagationConfig(
        name="primary_d10_scatter015_v4lambda_1m",
        frequency_hz=6.69e9,
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
        tx_vertical_spacing=4.0,
        tx_horizontal_spacing=0.5,
        max_depth=10,
        synthetic_array=True,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
        diffraction=False,
        edge_diffraction=False,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        source_code=1,
    ),
}


def prepare_scene_files(
    data_dir: Path | None = None,
    scene_dir: Path | None = None,
) -> Path:
    """Build the official-map XML and deterministic ground plane."""

    data_dir = Path(data_dir) if data_dir is not None else PATHS.round2
    scene_dir = Path(scene_dir) if scene_dir is not None else SCENE_DIR
    scene_xml = scene_dir / "round2_scene.xml"
    ground_obj = scene_dir / "ground.obj"
    scene_dir.mkdir(parents=True, exist_ok=True)
    map_path = data_dir / "Round2_Map.ply"
    positions = np.load(data_dir / "Round2_Train_Pos.npy")
    pad = 120.0
    x0 = float(positions[:, 0].min() - pad)
    x1 = float(positions[:, 0].max() + pad)
    y0 = float(positions[:, 1].min() - pad)
    y1 = float(positions[:, 1].max() + pad)
    ground = "\n".join(
        [
            "# Round 2 generated ground",
            f"v {x0:.6f} {y0:.6f} 0.0",
            f"v {x1:.6f} {y0:.6f} 0.0",
            f"v {x1:.6f} {y1:.6f} 0.0",
            f"v {x0:.6f} {y1:.6f} 0.0",
            "f 1 2 3 4",
            "",
        ]
    )
    relative_map = os.path.relpath(map_path, scene_dir)
    scene = "\n".join(
        [
            '<scene version="2.1.0">',
            '  <bsdf type="itu-radio-material" id="concrete">',
            '    <string name="type" value="concrete"/>',
            '    <float name="thickness" value="0.1"/>',
            "  </bsdf>",
            '  <shape type="ply" id="round2-map">',
            f'    <string name="filename" value="{relative_map}"/>',
            '    <boolean name="face_normals" value="true"/>',
            '    <ref id="concrete" name="bsdf"/>',
            "  </shape>",
            '  <shape type="obj" id="ground">',
            '    <string name="filename" value="ground.obj"/>',
            '    <boolean name="face_normals" value="true"/>',
            '    <ref id="concrete" name="bsdf"/>',
            "  </shape>",
            "</scene>",
            "",
        ]
    )
    if not ground_obj.is_file() or ground_obj.read_text(encoding="utf-8") != ground:
        ground_obj.write_text(ground, encoding="utf-8")
    if not scene_xml.is_file() or scene_xml.read_text(encoding="utf-8") != scene:
        scene_xml.write_text(scene, encoding="utf-8")
    return scene_xml


def select_drjit_cuda_device(device: int) -> None:
    library_path = Path(dr.__file__).with_name("libdrjit-core.so")
    library = ctypes.CDLL(str(library_path))
    setter = getattr(library, "_Z19jit_cuda_set_devicei")
    setter.argtypes = [ctypes.c_int]
    setter.restype = None
    setter(int(device))


def tx_target(position: np.ndarray, yaw_deg: float, tilt_deg: float):
    radius = 140.0
    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)
    return mi.Point3f(
        float(position[0] + radius * math.cos(yaw)),
        float(position[1] + radius * math.sin(yaw)),
        float(position[2] + radius * math.tan(tilt)),
    )


def configure_materials(scene, coefficient: float) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, material in scene.radio_materials.items():
        material.scattering_coefficient = coefficient
        value = np.asarray(material.scattering_coefficient.numpy()).reshape(-1)
        output[str(name)] = float(value[0])
    return output


def solver_seed(seed: int, site: int, split: str, first_point_id: int) -> int:
    return (
        int(seed)
        + 100_000 * int(site)
        + 1_000_000 * int(split == "test")
        + int(first_point_id)
    )


def _coefficients(paths) -> np.ndarray:
    raw = paths.a
    if isinstance(raw, tuple):
        real = np.asarray(raw[0].numpy(), dtype=np.float32)
        imag = np.asarray(raw[1].numpy(), dtype=np.float32)
        return (real + 1j * imag).astype(np.complex64)
    value = np.asarray(raw.numpy())
    if np.iscomplexobj(value):
        return value.astype(np.complex64)
    if value.ndim == 6 and value.shape[0] == 2:
        return (value[0] + 1j * value[1]).astype(np.complex64)
    raise ValueError(f"unexpected path coefficient representation {value.shape}")


def _mesh_geometry(scene) -> tuple[np.ndarray, np.ndarray]:
    shapes = scene.mi_scene.shapes()
    if len(shapes) != 1 or not isinstance(shapes[0], mi.Mesh):
        raise ValueError("physical source port expects one merged triangle mesh")
    mesh = shapes[0]
    vertices = dr.unravel(mi.Point3f, mesh.vertex_positions_buffer()).numpy().T
    faces = dr.unravel(mi.Point3u, mesh.faces_buffer()).numpy().T.astype(np.int64)
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(
        normals,
        np.maximum(lengths, 1e-30),
        out=np.zeros_like(normals),
    )
    return vertices.astype(np.float32), normals.astype(np.float32)


def extract_path_support(
    paths,
    scene,
    point_ids: np.ndarray,
    receiver_positions: np.ndarray,
    source_position: np.ndarray,
    config: PropagationConfig,
    seed: int,
) -> dict[str, np.ndarray]:
    """Convert solver output to compact physical support without retaining H."""

    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    tau = np.asarray(paths.tau.numpy(), dtype=np.float32)
    theta_t = np.asarray(paths.theta_t.numpy(), dtype=np.float32)
    phi_t = np.asarray(paths.phi_t.numpy(), dtype=np.float32)
    theta_r = np.asarray(paths.theta_r.numpy(), dtype=np.float32)
    phi_r = np.asarray(paths.phi_r.numpy(), dtype=np.float32)
    interactions = np.asarray(paths.interactions.numpy(), dtype=np.uint32)
    objects = np.asarray(paths.objects.numpy(), dtype=np.uint32)
    primitives = np.asarray(paths.primitives.numpy(), dtype=np.uint32)
    path_vertices = np.asarray(paths.vertices.numpy(), dtype=np.float32)
    coefficients = _coefficients(paths)
    _, face_normals = _mesh_geometry(scene)

    offsets = [0]
    banks: dict[str, list[np.ndarray]] = {
        "tau": [],
        "path_length_m": [],
        "geometric_length_m": [],
        "angles": [],
        "interactions": [],
        "objects": [],
        "primitives": [],
        "surface_plane": [],
        "vertices": [],
        "normals": [],
        "incidence_angle": [],
        "interaction_count": [],
        "absolute_energy": [],
        "relative_energy": [],
        "complex_coefficient": [],
        "full_complex_coefficient": [],
        "energy_rank": [],
        "geometry_valid": [],
        "seed": [],
        "source": [],
    }
    max_depth = interactions.shape[0]

    for local_index in range(len(point_ids)):
        indices = np.flatnonzero(valid[local_index, 0])
        count = len(indices)
        offsets.append(offsets[-1] + count)
        if count == 0:
            continue
        local_interactions = np.take(
            interactions[:, local_index, 0, :], indices, axis=-1
        ).T.astype(np.uint8)
        local_objects = np.take(
            objects[:, local_index, 0, :], indices, axis=-1
        ).T.astype(np.uint32)
        local_primitives = np.take(
            primitives[:, local_index, 0, :], indices, axis=-1
        ).T.astype(np.uint32)
        local_vertices = (
            np.take(path_vertices[:, local_index, 0, :, :], indices, axis=1)
            .transpose(1, 0, 2)
            .astype(np.float32)
        )
        interaction_count = np.count_nonzero(local_interactions, axis=1).astype(
            np.uint8
        )

        local_coefficients = np.take(
            coefficients[local_index, :, 0, :, :], indices, axis=-1
        ).transpose(2, 1, 0)
        local_coefficients = local_coefficients.reshape(count, 2, 16, 8, 4).astype(
            np.complex64
        )
        reference_coefficients = local_coefficients[:, :, 0, 0, :].reshape(count, 8)
        if reference_coefficients.shape[1] != 8:
            raise ValueError(
                "expected 8 physical path ports, got " f"{reference_coefficients.shape}"
            )
        energy = np.square(np.abs(local_coefficients), dtype=np.float32).sum(
            axis=(1, 2, 3, 4), dtype=np.float64
        )
        relative_energy = (
            energy / max(float(energy.sum(dtype=np.float64)), 1e-30)
        ).astype(np.float32)
        order = np.argsort(-relative_energy, kind="stable")
        rank = np.empty(count, dtype=np.uint32)
        rank[order] = np.arange(count, dtype=np.uint32)

        local_normals = np.zeros((count, max_depth, 3), dtype=np.float32)
        surface_plane = np.zeros((count, max_depth, 4), dtype=np.int32)
        incidence = np.zeros((count, max_depth), dtype=np.float32)
        geometric_length = np.zeros(count, dtype=np.float64)
        geometry_valid = np.ones(count, dtype=bool)
        receiver = np.asarray(receiver_positions[local_index], dtype=np.float64)
        source = np.asarray(source_position, dtype=np.float64)
        for path_index in range(count):
            depth = int(interaction_count[path_index])
            previous = source
            for interaction_index in range(depth):
                primitive = int(local_primitives[path_index, interaction_index])
                if primitive >= len(face_normals):
                    geometry_valid[path_index] = False
                    continue
                normal = np.asarray(face_normals[primitive], dtype=np.float64)
                vertex = np.asarray(
                    local_vertices[path_index, interaction_index], dtype=np.float64
                )
                nonzero = np.flatnonzero(np.abs(normal) > 1e-8)
                if len(nonzero) and normal[nonzero[0]] < 0.0:
                    normal = -normal
                plane_offset = float(np.dot(normal, vertex))
                surface_plane[path_index, interaction_index, :3] = np.rint(
                    normal * 100_000.0
                ).astype(np.int32)
                surface_plane[path_index, interaction_index, 3] = int(
                    np.rint(plane_offset * 100.0)
                )
                segment = vertex - previous
                segment_length = float(np.linalg.norm(segment))
                if segment_length <= 0.0 or not np.isfinite(segment_length):
                    geometry_valid[path_index] = False
                else:
                    direction = segment / segment_length
                    cosine = float(np.clip(abs(np.dot(-direction, normal)), 0.0, 1.0))
                    incidence[path_index, interaction_index] = math.acos(cosine)
                    geometric_length[path_index] += segment_length
                local_normals[path_index, interaction_index] = normal
                previous = vertex
            final_length = float(np.linalg.norm(receiver - previous))
            geometric_length[path_index] += final_length
            geometry_valid[path_index] &= np.isfinite(final_length)

        local_tau = tau[local_index, 0, indices]
        banks["tau"].append(local_tau)
        banks["path_length_m"].append(
            (local_tau.astype(np.float64) * speed_of_light).astype(np.float32)
        )
        banks["geometric_length_m"].append(geometric_length.astype(np.float32))
        banks["angles"].append(
            np.stack(
                [
                    theta_t[local_index, 0, indices],
                    phi_t[local_index, 0, indices],
                    theta_r[local_index, 0, indices],
                    phi_r[local_index, 0, indices],
                ],
                axis=-1,
            ).astype(np.float32)
        )
        banks["interactions"].append(local_interactions)
        banks["objects"].append(local_objects)
        banks["primitives"].append(local_primitives)
        banks["surface_plane"].append(surface_plane)
        banks["vertices"].append(local_vertices)
        banks["normals"].append(local_normals)
        banks["incidence_angle"].append(incidence)
        banks["interaction_count"].append(interaction_count)
        banks["absolute_energy"].append(energy.astype(np.float32))
        banks["relative_energy"].append(relative_energy)
        banks["complex_coefficient"].append(reference_coefficients)
        banks["full_complex_coefficient"].append(
            local_coefficients.reshape(count, 256, 4)
        )
        banks["energy_rank"].append(rank)
        banks["geometry_valid"].append(geometry_valid)
        banks["seed"].append(np.full(count, seed, dtype=np.int32))
        banks["source"].append(np.full(count, config.source_code, dtype=np.int8))

    empty_shapes = {
        "tau": (0,),
        "path_length_m": (0,),
        "geometric_length_m": (0,),
        "angles": (0, 4),
        "interactions": (0, max_depth),
        "objects": (0, max_depth),
        "primitives": (0, max_depth),
        "surface_plane": (0, max_depth, 4),
        "vertices": (0, max_depth, 3),
        "normals": (0, max_depth, 3),
        "incidence_angle": (0, max_depth),
        "interaction_count": (0,),
        "absolute_energy": (0,),
        "relative_energy": (0,),
        "complex_coefficient": (0, 8),
        "full_complex_coefficient": (0, 256, 4),
        "energy_rank": (0,),
        "geometry_valid": (0,),
        "seed": (0,),
        "source": (0,),
    }
    dtypes = {
        "complex_coefficient": np.complex64,
        "full_complex_coefficient": np.complex64,
        "interactions": np.uint8,
        "objects": np.uint32,
        "primitives": np.uint32,
        "surface_plane": np.int32,
        "interaction_count": np.uint8,
        "energy_rank": np.uint32,
        "geometry_valid": bool,
        "seed": np.int32,
        "source": np.int8,
    }
    output: dict[str, np.ndarray] = {
        "point_ids": np.asarray(point_ids, dtype=np.int64),
        "offsets": np.asarray(offsets, dtype=np.int64),
    }
    for name, parts in banks.items():
        output[name] = (
            np.concatenate(parts, axis=0)
            if parts
            else np.empty(empty_shapes[name], dtype=dtypes.get(name, np.float32))
        )
    return output


def metadata(
    backend: str,
    config: PropagationConfig,
    seed: int,
    site: int,
    split: str,
    point_ids: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "configuration": asdict(config),
        "seed": int(seed),
        "site": int(site),
        "split": split,
        "point_ids": np.asarray(point_ids, dtype=np.int64).tolist(),
        "elapsed_seconds": float(elapsed_seconds),
        "mitsuba_version": mi.__version__,
        "drjit_version": dr.__version__,
        "mitsuba_variant": mi.variant(),
    }


def save_support(
    path: Path, support: dict[str, np.ndarray], meta: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **support,
        metadata=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    os.replace(temporary, path)
