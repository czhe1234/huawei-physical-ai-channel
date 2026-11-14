"""Frozen production configuration for the submission pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SEEDS = (
    42,
    101,
    211,
    307,
    401,
    503,
    601,
    701,
    809,
    907,
    1009,
    1103,
    1201,
    1301,
    1409,
    1511,
    1601,
    1709,
    1801,
    1901,
    2003,
    2111,
    2203,
    2309,
)


@dataclass(frozen=True)
class PipelineConfig:
    seed_count: int = 24
    receiver_batch_size: int = 16
    physical_carrier_frequency_hz: float = 6.69e9
    paths_per_point: int = 128
    base_reserve: int = 48
    base_paths_per_seed: int = 96
    scatter_paths_per_seed: int = 64
    scatter_pool: int = 256
    inference_batch_size: int = 64
    pas_blend: float = 0.20
    pdp_blend: float = 0.20
    projection_cycles: int = 4
    subcarriers: int = 192
    subcarrier_spacing_hz: float = 30_000.0
    tx_polarizations: int = 2
    tx_horizontal: int = 16
    tx_vertical: int = 8
    rx_ports: int = 4
    interaction_steps: int = 16

    @property
    def seeds(self) -> tuple[int, ...]:
        if not 1 <= self.seed_count <= len(SEEDS):
            raise ValueError(f"seed_count must be in [1, {len(SEEDS)}]")
        return SEEDS[: self.seed_count]


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    data_dir: Path
    work_dir: Path
    model_dir: Path
    output: Path

    @classmethod
    def create(
        cls,
        project_root: Path,
        data_dir: Path | None = None,
        work_dir: Path | None = None,
        model_dir: Path | None = None,
        output: Path | None = None,
    ) -> "RuntimePaths":
        root = project_root.resolve()
        return cls(
            project_root=root,
            data_dir=(data_dir or root / "datasets" / "round2").resolve(),
            work_dir=(
                work_dir or root / "artifacts" / "round2" / "physical_ai_submission"
            ).resolve(),
            model_dir=(
                model_dir
                or root / "artifacts" / "round2" / "physical_ai_submission" / "models"
            ).resolve(),
            output=(output or root / "Round2_Test_Channel.npy").resolve(),
        )

    @property
    def scene_dir(self) -> Path:
        return self.work_dir / "scene"

    @property
    def propagation_dir(self) -> Path:
        return self.work_dir / "propagation"

    @property
    def token_dir(self) -> Path:
        return self.work_dir / "path_tokens"

    @property
    def base_dir(self) -> Path:
        return self.work_dir / "base_field"

    @property
    def report_dir(self) -> Path:
        return self.work_dir / "reports"
