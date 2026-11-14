"""Frozen configuration for the source-aligned propagation implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceBranchConfig:
    name: str
    source: int
    max_depth: int
    scattering_coefficient: float
    rays_per_site_seed: int = 1_000_000
    shard_size: int = 250_000


@dataclass(frozen=True)
class SourceAlignedConfig:
    frequency_hz: float = 6.69e9
    site_yaw_deg: tuple[float, float] = (-44.0, 73.5)
    site_tilt_deg: tuple[float, float] = (-11.5, -10.5)
    seeds: tuple[int, ...] = (42, 101, 211)
    base: SourceBranchConfig = field(
        default_factory=lambda: SourceBranchConfig(
            name="base_d16",
            source=0,
            max_depth=16,
            scattering_coefficient=0.0,
        )
    )
    scatter: SourceBranchConfig = field(
        default_factory=lambda: SourceBranchConfig(
            name="scatter015_d10",
            source=1,
            max_depth=10,
            scattering_coefficient=0.15,
        )
    )
    concrete_thickness_m: float = 0.1
    material_table_samples: int = 16_385
    ray_epsilon_m: float = 1e-4
    routes_per_depth: int = 1_000_000
    maximum_routes: int = 1_000_000
    scatter_records_per_seed: int = 1_000_000
    scatter_records_per_voxel: int = 8
    scatter_voxel_m: float = 1.0
    scatter_solve_pool: int = 3_000_000
    raw_candidate_count: int = 512
    base_pool: int = 256
    scatter_pool: int = 256
    target_batch_size: int = 16
    maximum_interactions: int = 16

    @property
    def branches(self) -> tuple[SourceBranchConfig, SourceBranchConfig]:
        return self.base, self.scatter

    def payload(self) -> dict[str, Any]:
        return asdict(self)
