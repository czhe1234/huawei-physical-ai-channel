#!/usr/bin/env python3
"""Configuration definitions for the Round 2 E200 Sionna search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import sys
from typing import Any

from project_paths import PATHS


def portable_text(value: str) -> str:
    """Replace machine-local prefixes while preserving report diagnostics."""
    replacements = (
        (str(PATHS.root), "<PROJECT_ROOT>"),
        (os.environ.get("CONDA_PREFIX", ""), "<CONDA_PREFIX>"),
        (sys.prefix, "<PYTHON_PREFIX>"),
        (str(Path.home()), "<HOME>"),
    )
    seen: set[str] = set()
    for source, replacement in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if not source or source == "/" or source in seen:
            continue
        value = value.replace(source, replacement)
        seen.add(source)
    return value


def portable_payload(value: Any) -> Any:
    """Recursively make JSON report strings independent of the current host."""
    if isinstance(value, dict):
        return {key: portable_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(portable_payload(item) for item in value)
    if isinstance(value, Path):
        return portable_text(str(value))
    if isinstance(value, str):
        return portable_text(value)
    return value


@dataclass(frozen=True)
class SearchConfig:
    name: str
    family: str
    frequency_hz: float = 6.690e9
    site_yaw_deg: tuple[float, float] = (-43.0, 74.0)
    site_tilt_deg: tuple[float, float] = (-11.0, -10.0)
    tx_vertical_spacing: float = 4.0
    tx_horizontal_spacing: float = 0.5
    rx_orientation: str = "track_bs"
    max_depth: int = 16
    synthetic_array: bool = True
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True
    diffraction: bool = False
    edge_diffraction: bool = False
    building_material: str = "concrete"
    ground_material: str = "concrete"
    building_scattering: float = 0.0
    ground_scattering: float = 0.0
    samples_per_src: int = 1_000_000
    max_num_paths_per_src: int = 1_000_000
    seed: int = 42
    parent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SearchConfig":
        payload = dict(value)
        payload["site_yaw_deg"] = tuple(payload["site_yaw_deg"])
        payload["site_tilt_deg"] = tuple(payload["site_tilt_deg"])
        return cls(**payload)


def geometry_configs() -> list[SearchConfig]:
    configs: list[SearchConfig] = []

    def add(name: str, **changes: Any) -> None:
        configs.append(SearchConfig(name=name, family="geometry", **changes))

    add(
        "g00_old_control",
        site_yaw_deg=(-44.0, 73.5),
        site_tilt_deg=(-11.5, -10.5),
    )
    add("g01_geometric_center")

    tilt_pairs = (
        (-18.0, -18.0),
        (-18.0, -8.0),
        (-8.0, -18.0),
        (-8.0, -8.0),
        (-3.0, -3.0),
        (-13.0, -6.0),
    )
    for index, tilt in enumerate(tilt_pairs, start=2):
        add(f"g{index:02d}_tilt_{tilt[0]:+03.0f}_{tilt[1]:+03.0f}", site_tilt_deg=tilt)

    index = 8
    for yaw0 in (-58.0, -43.0, -28.0):
        for yaw1 in (59.0, 74.0, 89.0):
            add(
                f"g{index:02d}_yaw_{yaw0:+03.0f}_{yaw1:+03.0f}",
                site_yaw_deg=(yaw0, yaw1),
                site_tilt_deg=(-10.0, -9.0),
            )
            index += 1

    array_variants = (
        {"tx_vertical_spacing": 0.5},
        {"tx_vertical_spacing": 1.0},
        {"tx_vertical_spacing": 2.0},
        {"tx_vertical_spacing": 3.0},
        {"tx_vertical_spacing": 5.5},
        {"tx_horizontal_spacing": 0.4},
        {"tx_horizontal_spacing": 0.6},
        {"synthetic_array": False},
        {"rx_orientation": "fixed"},
        {"frequency_hz": 6.688e9},
        {"frequency_hz": 6.693e9},
    )
    for changes in array_variants:
        label = "_".join(f"{key}{value}" for key, value in changes.items())
        add(f"g{index:02d}_{label}", **changes)
        index += 1
    return configs


def physics_configs(parents: list[SearchConfig]) -> list[SearchConfig]:
    variants: tuple[tuple[str, dict[str, Any]], ...] = (
        ("base_d8", {"max_depth": 8}),
        ("base_d12", {"max_depth": 12}),
        ("base_d16", {"max_depth": 16}),
        ("base_d24", {"max_depth": 24}),
        ("edge_d8", {"max_depth": 8, "diffraction": True, "edge_diffraction": True}),
        ("edge_d12", {"max_depth": 12, "diffraction": True, "edge_diffraction": True}),
        (
            "diffuse_d8_s005",
            {"max_depth": 8, "diffuse_reflection": True, "building_scattering": 0.05},
        ),
        (
            "diffuse_d10_s015",
            {"max_depth": 10, "diffuse_reflection": True, "building_scattering": 0.15},
        ),
        (
            "diffuse_d16_s015",
            {"max_depth": 16, "diffuse_reflection": True, "building_scattering": 0.15},
        ),
        (
            "diffuse_d10_s030",
            {"max_depth": 10, "diffuse_reflection": True, "building_scattering": 0.30},
        ),
        (
            "diffuse_edge_d8_s015",
            {
                "max_depth": 8,
                "diffuse_reflection": True,
                "building_scattering": 0.15,
                "diffraction": True,
                "edge_diffraction": True,
            },
        ),
        ("ground_very_dry", {"ground_material": "very_dry_ground"}),
        ("ground_medium_dry", {"ground_material": "medium_dry_ground"}),
        ("ground_wet", {"ground_material": "wet_ground"}),
        (
            "brick_medium_ground",
            {"building_material": "brick", "ground_material": "medium_dry_ground"},
        ),
        (
            "glass_medium_ground",
            {"building_material": "glass", "ground_material": "medium_dry_ground"},
        ),
        (
            "marble_medium_ground",
            {"building_material": "marble", "ground_material": "medium_dry_ground"},
        ),
        (
            "wood_medium_ground",
            {"building_material": "wood", "ground_material": "medium_dry_ground"},
        ),
    )
    output: list[SearchConfig] = []
    for parent_index, parent in enumerate(parents):
        for variant, changes in variants:
            output.append(
                replace(
                    parent,
                    name=f"p{parent_index}_{variant}_{parent.name}",
                    family="physics",
                    parent=parent.name,
                    **changes,
                )
            )
    return output


def config_signature(config: SearchConfig) -> tuple[Any, ...]:
    return (
        config.site_yaw_deg,
        config.site_tilt_deg,
        config.tx_vertical_spacing,
        config.tx_horizontal_spacing,
        config.rx_orientation,
        config.max_depth,
        config.synthetic_array,
        config.diffuse_reflection,
        config.diffraction,
        config.edge_diffraction,
        config.building_material,
        config.ground_material,
        config.building_scattering,
        config.frequency_hz,
    )
