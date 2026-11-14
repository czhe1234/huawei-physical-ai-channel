from __future__ import annotations

import time
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from round2.physical_prior.geometry_prior_engine import (
    ScenePriorBuilder,
    build_receive_array,
    build_transmit_array,
    Source,
    Target,
    load_scene,
)
from round2.physical_prior.scene_prior_data import (
    CONFIGS,
    MAX_COMPONENTS_PER_SOURCE,
    SAMPLES_PER_SOURCE,
    SCENE_FREQUENCY_HZ,
    SITE_TILT_DEG,
    SITE_YAW_DEG,
    TX_HORIZONTAL_SPACING,
    TX_VERTICAL_SPACING,
    ScenePriorConfig,
    configure_scene_response,
    extract_component_support,
    metadata,
    prepare_scene_files,
    select_drjit_cuda_device,
    support_sampling_seed,
    tx_target,
)


class ScenePriorSession:
    def __init__(
        self,
        *,
        site: int,
        bs_position: np.ndarray,
        config: str | ScenePriorConfig,
        seed: int,
        cuda_device: int = 0,
        data_dir: Path,
        scene_dir: Path,
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
            str(prepare_scene_files(data_dir=data_dir, scene_dir=scene_dir))
        )
        self.scene.frequency = SCENE_FREQUENCY_HZ
        self.scene.tx_array = build_transmit_array(
            TX_VERTICAL_SPACING,
            TX_HORIZONTAL_SPACING,
        )
        self.scene.rx_array = build_receive_array()
        configure_scene_response(self.scene, config.scattering_coefficient)
        self.transmitter = Source(
            name=f"tx_site{self .site }_seed{self .seed }",
            position=mi.Point3f(*map(float, self.bs_position)),
            look_at=tx_target(
                self.bs_position,
                SITE_YAW_DEG[self.site],
                SITE_TILT_DEG[self.site],
            ),
        )
        self.scene.add(self.transmitter)
        self.prior_builder = ScenePriorBuilder()
        self.setup_seconds = float(time.time() - started)

    def build_scene_prior(
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
            raise ValueError(f"unsupported split: {split }")

        started = time.time()
        for name in self.receiver_names:
            self.scene.remove(name)
        self.receiver_names = []
        endpoint_started = time.time()
        for point_id, position in zip(point_ids, positions):
            name = f"rx_{split }_{int (point_id ):05d}"
            self.scene.add(
                Target(
                    name=name,
                    position=mi.Point3f(*map(float, position)),
                    look_at=self.transmitter,
                )
            )
            self.receiver_names.append(name)
        endpoint_finished = time.time()

        builder_started = time.time()
        components = self.prior_builder(
            self.scene,
            max_depth=self.config.max_depth,
            max_components_per_source=MAX_COMPONENTS_PER_SOURCE,
            samples_per_src=SAMPLES_PER_SOURCE,
            diffuse_reflection=self.config.diffuse_reflection,
            seed=support_sampling_seed(self.seed, self.site, split, int(point_ids[0])),
        )
        builder_finished = time.time()
        support = extract_component_support(
            components,
            point_ids,
            self.seed,
        )
        extraction_finished = time.time()
        del components
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
            "prior_builder_seconds": builder_finished - builder_started,
            "support_extraction_seconds": extraction_finished - builder_finished,
        }
        return support, report

    def close(self) -> None:
        for name in self.receiver_names:
            self.scene.remove(name)
        self.receiver_names = []
        dr.sync_thread()

__all__ = [
    "CONFIGS",
    "ScenePriorSession",
    "ScenePriorConfig",
]
