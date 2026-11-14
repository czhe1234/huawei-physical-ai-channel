"""Production entry point for compact physical path support."""

from __future__ import annotations

import time
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from round2.physical_prior.physical_rt import (
    PathSolver,
    PlanarArray,
    Receiver,
    Transmitter,
    load_scene,
)
from round2.physical_prior.source_port_common import (
    CONFIGS,
    PropagationConfig,
    configure_materials,
    extract_path_support,
    metadata,
    prepare_scene_files,
    select_drjit_cuda_device,
    solver_seed,
    tx_target,
)


class PathSupportSession:
    """Keep the scene and JIT-compiled solver resident across receiver batches."""

    def __init__(
        self,
        *,
        site: int,
        bs_position: np.ndarray,
        config: str | PropagationConfig,
        seed: int,
        cuda_device: int = 0,
        data_dir: Path | None = None,
        scene_dir: Path | None = None,
    ) -> None:
        if isinstance(config, str):
            config = CONFIGS[config]
        self.site = int(site)
        self.bs_position = np.asarray(bs_position, dtype=np.float32)
        self.config = config
        self.seed = int(seed)
        self.cuda_device = int(cuda_device)
        self.receiver_names: list[str] = []

        started = time.time()
        select_drjit_cuda_device(self.cuda_device)
        self.scene = load_scene(
            str(prepare_scene_files(data_dir=data_dir, scene_dir=scene_dir)),
            merge_shapes=True,
            remove_duplicate_vertices=True,
        )
        self.scene.frequency = config.frequency_hz
        self.scene.tx_array = PlanarArray(
            num_rows=8,
            num_cols=16,
            vertical_spacing=config.tx_vertical_spacing,
            horizontal_spacing=config.tx_horizontal_spacing,
            pattern="tr38901",
            polarization="VH",
        )
        self.scene.rx_array = PlanarArray(
            num_rows=2,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="cross",
        )
        configure_materials(self.scene, config.scattering_coefficient)
        self.transmitter = Transmitter(
            name=f"tx_site{self.site}_seed{self.seed}",
            position=mi.Point3f(*map(float, self.bs_position)),
            look_at=tx_target(
                self.bs_position,
                config.site_yaw_deg[self.site],
                config.site_tilt_deg[self.site],
            ),
        )
        self.scene.add(self.transmitter)
        self.solver = PathSolver()
        self.setup_seconds = float(time.time() - started)

    def solve(
        self,
        *,
        positions: np.ndarray,
        point_ids: np.ndarray,
        split: str = "train",
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        point_ids = np.asarray(point_ids, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.float32)
        if len(point_ids) != len(positions):
            raise ValueError("point_ids and positions must have the same length")
        if len(point_ids) == 0:
            raise ValueError("at least one receiver is required")
        if split not in ("train", "test"):
            raise ValueError(f"unsupported split: {split}")

        started = time.time()
        for name in self.receiver_names:
            self.scene.remove(name)
        self.receiver_names = []
        endpoint_started = time.time()
        for point_id, position in zip(point_ids, positions):
            name = f"rx_{split}_{int(point_id):05d}"
            self.scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*map(float, position)),
                    look_at=self.transmitter,
                )
            )
            self.receiver_names.append(name)
        endpoint_finished = time.time()

        solver_started = time.time()
        paths = self.solver(
            self.scene,
            max_depth=self.config.max_depth,
            max_num_paths_per_src=self.config.max_num_paths_per_src,
            samples_per_src=self.config.samples_per_src,
            los=True,
            specular_reflection=True,
            diffuse_reflection=self.config.diffuse_reflection,
            refraction=True,
            seed=solver_seed(self.seed, self.site, split, int(point_ids[0])),
        )
        solver_finished = time.time()
        support = extract_path_support(
            paths,
            self.scene,
            point_ids,
            positions,
            self.bs_position,
            self.config,
            self.seed,
        )
        extraction_finished = time.time()
        del paths
        dr.sync_thread()

        report = metadata(
            "port",
            self.config,
            self.seed,
            self.site,
            split,
            point_ids,
            extraction_finished - started,
        )
        report["cuda_device"] = self.cuda_device
        report["timings"] = {
            "resident_scene_setup_seconds": self.setup_seconds,
            "endpoint_setup_seconds": endpoint_finished - endpoint_started,
            "path_solver_seconds": solver_finished - solver_started,
            "support_extraction_seconds": extraction_finished - solver_finished,
        }
        return support, report

    def close(self) -> None:
        for name in self.receiver_names:
            self.scene.remove(name)
        self.receiver_names = []
        dr.sync_thread()


def run_path_support(
    *,
    positions: np.ndarray,
    point_ids: np.ndarray,
    site: int,
    bs_position: np.ndarray,
    config: str | PropagationConfig,
    seed: int,
    split: str = "train",
    cuda_device: int = 0,
    data_dir: Path | None = None,
    scene_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Generate compact path support without returning a channel tensor."""

    session = PathSupportSession(
        site=site,
        bs_position=bs_position,
        config=config,
        seed=seed,
        cuda_device=cuda_device,
        data_dir=data_dir,
        scene_dir=scene_dir,
    )
    try:
        return session.solve(
            positions=positions,
            point_ids=point_ids,
            split=split,
        )
    finally:
        session.close()


__all__ = [
    "CONFIGS",
    "PathSupportSession",
    "PropagationConfig",
    "run_path_support",
]
