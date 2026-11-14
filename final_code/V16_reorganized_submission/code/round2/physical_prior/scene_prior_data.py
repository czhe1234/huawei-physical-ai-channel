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

SCENE_FREQUENCY_HZ = 6.69e9
SITE_YAW_DEG = (-44.0, 73.5)
SITE_TILT_DEG = (-11.5, -10.5)
TX_VERTICAL_SPACING = 4.0
TX_HORIZONTAL_SPACING = 0.5
SAMPLES_PER_SOURCE = 1_000_000
MAX_COMPONENTS_PER_SOURCE = 1_000_000


@dataclass(frozen=True)
class ScenePriorConfig:
    max_depth: int
    diffuse_reflection: bool
    scattering_coefficient: float


CONFIGS = {
    "base": ScenePriorConfig(
        max_depth=16,
        diffuse_reflection=False,
        scattering_coefficient=0.0,
    ),
    "scatter": ScenePriorConfig(
        max_depth=10,
        diffuse_reflection=True,
        scattering_coefficient=0.15,
    ),
}


def prepare_scene_files(
    data_dir: Path,
    scene_dir: Path,
) -> Path:
    data_dir = Path(data_dir)
    scene_dir = Path(scene_dir)
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
            f"v {x0 :.6f} {y0 :.6f} 0.0",
            f"v {x1 :.6f} {y0 :.6f} 0.0",
            f"v {x1 :.6f} {y1 :.6f} 0.0",
            f"v {x0 :.6f} {y1 :.6f} 0.0",
            "f 1 2 3 4",
            "",
        ]
    )
    relative_map = os.path.relpath(map_path, scene_dir)
    scene = "\n".join(
        [
            '<scene version="2.1.0">',
            '  <bsdf type="scene-interaction-response" id="scene-interaction">',
            '    <float name="thickness" value="0.1"/>',
            "  </bsdf>",
            '  <shape type="ply" id="round2-map">',
            f'    <string name="filename" value="{relative_map }"/>',
            '    <boolean name="face_normals" value="true"/>',
            '    <ref id="scene-interaction" name="bsdf"/>',
            "  </shape>",
            '  <shape type="obj" id="ground">',
            '    <string name="filename" value="ground.obj"/>',
            '    <boolean name="face_normals" value="true"/>',
            '    <ref id="scene-interaction" name="bsdf"/>',
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


def configure_scene_response(scene, coefficient: float) -> None:
    for response in scene.interaction_responses.values():
        response.scattering_coefficient = coefficient


def support_sampling_seed(seed: int, site: int, split: str, first_point_id: int) -> int:
    return (
        int(seed)
        + 100_000 * int(site)
        + 1_000_000 * int(split == "test")
        + int(first_point_id)
    )


def _component_coefficients(components) -> np.ndarray:
    raw = components.a
    if isinstance(raw, tuple):
        real = np.asarray(raw[0].numpy(), dtype=np.float32)
        imag = np.asarray(raw[1].numpy(), dtype=np.float32)
        return (real + 1j * imag).astype(np.complex64)
    value = np.asarray(raw.numpy())
    if np.iscomplexobj(value):
        return value.astype(np.complex64)
    if value.ndim == 6 and value.shape[0] == 2:
        return (value[0] + 1j * value[1]).astype(np.complex64)
    raise ValueError(f"unexpected component coefficient representation {value .shape }")


def extract_component_support(
    components,
    point_ids: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    valid = np.asarray(components.valid.numpy(), dtype=bool)
    tau = np.asarray(components.tau.numpy(), dtype=np.float32)
    theta_t = np.asarray(components.theta_t.numpy(), dtype=np.float32)
    phi_t = np.asarray(components.phi_t.numpy(), dtype=np.float32)
    theta_r = np.asarray(components.theta_r.numpy(), dtype=np.float32)
    phi_r = np.asarray(components.phi_r.numpy(), dtype=np.float32)
    interactions = np.asarray(components.interactions.numpy(), dtype=np.uint32)
    objects = np.asarray(components.objects.numpy(), dtype=np.uint32)
    primitives = np.asarray(components.primitives.numpy(), dtype=np.uint32)
    interaction_vertices = np.asarray(components.vertices.numpy(), dtype=np.float32)
    coefficients = _component_coefficients(components)

    offsets = [0]
    banks: dict[str, list[np.ndarray]] = {
        "tau": [],
        "angles": [],
        "interactions": [],
        "objects": [],
        "primitives": [],
        "vertices": [],
        "interaction_count": [],
        "absolute_energy": [],
        "relative_energy": [],
        "full_complex_coefficient": [],
        "seed": [],
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
            np.take(interaction_vertices[:, local_index, 0, :, :], indices, axis=1)
            .transpose(1, 0, 2)
            .astype(np.float32)
        )
        interaction_count = np.count_nonzero(local_interactions, axis=1).astype(
            np.uint8
        )

        local_component_coefficients = np.take(
            coefficients[local_index, :, 0, :, :], indices, axis=-1
        ).transpose(2, 1, 0)
        local_component_coefficients = local_component_coefficients.reshape(count, 2, 16, 8, 4).astype(
            np.complex64
        )
        energy = np.square(np.abs(local_component_coefficients), dtype=np.float32).sum(
            axis=(1, 2, 3, 4), dtype=np.float64
        )
        relative_energy = (
            energy / max(float(energy.sum(dtype=np.float64)), 1e-30)
        ).astype(np.float32)

        local_tau = tau[local_index, 0, indices]
        banks["tau"].append(local_tau)
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
        banks["vertices"].append(local_vertices)
        banks["interaction_count"].append(interaction_count)
        banks["absolute_energy"].append(energy.astype(np.float32))
        banks["relative_energy"].append(relative_energy)
        banks["full_complex_coefficient"].append(
            local_component_coefficients.reshape(count, 256, 4)
        )
        banks["seed"].append(np.full(count, seed, dtype=np.int32))

    empty_shapes = {
        "tau": (0,),
        "angles": (0, 4),
        "interactions": (0, max_depth),
        "objects": (0, max_depth),
        "primitives": (0, max_depth),
        "vertices": (0, max_depth, 3),
        "interaction_count": (0,),
        "absolute_energy": (0,),
        "relative_energy": (0,),
        "full_complex_coefficient": (0, 256, 4),
        "seed": (0,),
    }
    dtypes = {
        "full_complex_coefficient": np.complex64,
        "interactions": np.uint8,
        "objects": np.uint32,
        "primitives": np.uint32,
        "interaction_count": np.uint8,
        "seed": np.int32,
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
    config: ScenePriorConfig,
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
