"""Frozen configuration for the Round 2 physical-prior module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


LIGHT_SPEED_M_S = 299_792_458.0
SUBCARRIER_SPACING_HZ = 30_000.0
DELAY_BINS = 192
HORIZONTAL_BINS = 16
VERTICAL_BINS = 8
METERS_PER_DELAY_BIN = LIGHT_SPEED_M_S / (
    SUBCARRIER_SPACING_HZ * DELAY_BINS
)

INTERACTION_REFLECTION = 1
INTERACTION_SCATTER = 2
INTERACTION_TRANSMISSION = 4

SOURCE_BASE = 0
SOURCE_SCATTER = 1


@dataclass(frozen=True)
class BranchConfig:
    name: str
    source: int
    max_depth: int
    diffuse_probability: float
    rays_per_site_seed: int = 1_000_000
    shard_size: int = 250_000


@dataclass(frozen=True)
class PhysicalPriorConfig:
    """The user-approved three-seed E185-equivalent information contract."""

    seeds: tuple[int, ...] = (42, 101, 211)
    base: BranchConfig = field(
        default_factory=lambda: BranchConfig(
            name="base",
            source=SOURCE_BASE,
            max_depth=16,
            diffuse_probability=0.0,
        )
    )
    scatter: BranchConfig = field(
        default_factory=lambda: BranchConfig(
            name="scatter015",
            source=SOURCE_SCATTER,
            max_depth=10,
            diffuse_probability=0.15,
        )
    )
    maximum_path_length_m: float = 1_200.0
    ray_epsilon_m: float = 0.002
    minimum_ray_energy: float = 1e-7
    concrete_thickness_m: float = 0.1
    concrete_relative_permittivity: float = 5.24
    concrete_conductivity_s_m: float = 0.0462
    ground_padding_m: float = 120.0
    family_plane_tolerance: float = 2e-3
    family_bounds_margin_m: float = 0.08
    route_bank_per_depth: int = 512
    route_bank_maximum: int = 8_192
    scatter_events_per_seed: int = 65_536
    scatter_events_per_voxel: int = 4
    scatter_voxel_m: float = 1.0
    base_per_seed: int = 96
    scatter_per_seed: int = 64
    base_pool: int = 256
    scatter_pool: int = 256
    raw_candidate_count: int = 512
    output_candidate_count: int = 128
    base_reserve: int = 48
    maximum_interactions: int = 16
    horizontal_scale: float = 1.5
    vertical_scale: float = 1.0
    delay_scale: float = 2.0
    match_cost_limit: float = 1.0
    match_bandwidth: float = 0.25
    minimum_truth_fraction: float = 1e-5
    rank_hidden: int = 128
    rank_sequence_hidden: int = 64
    rank_epochs: int = 18
    rank_batch_points: int = 24
    rank_learning_rate: float = 8e-4
    rank_weight_decay: float = 2e-5
    rank_positive_weight: float = 8.0
    rank_diversity_cost: float = 0.18
    ranking_seed: int = 20_260_814_206

    @property
    def branches(self) -> tuple[BranchConfig, BranchConfig]:
        return self.base, self.scatter

    def payload(self) -> dict[str, Any]:
        return asdict(self)
