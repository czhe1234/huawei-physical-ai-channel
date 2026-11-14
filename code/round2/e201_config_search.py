#!/usr/bin/env python3
"""Configuration definitions for the adaptive Round 2 E201 Etoile search."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from e200_config_search import SearchConfig, config_signature, portable_payload


def baseline() -> SearchConfig:
    return SearchConfig(
        name="b1_base",
        family="direction_batch1",
        frequency_hz=6.690e9,
        site_yaw_deg=(-43.0, 74.0),
        site_tilt_deg=(-8.0, -8.0),
        tx_vertical_spacing=4.0,
        tx_horizontal_spacing=0.5,
        rx_orientation="track_bs",
        max_depth=16,
        synthetic_array=True,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        edge_diffraction=False,
        building_material="scene_default",
        ground_material="scene_default",
        building_scattering=0.0,
        ground_scattering=0.0,
        samples_per_src=1_000_000,
        max_num_paths_per_src=1_000_000,
        seed=42,
    )


def batch1_configs() -> list[SearchConfig]:
    base = baseline()
    output = [base]

    def add(name: str, **changes: Any) -> None:
        output.append(replace(base, name=name, **changes))

    for value in (-63.0, -53.0, -33.0, -23.0):
        add(f"b1_yaw0_{value:+04.0f}", site_yaw_deg=(value, 74.0))
    for value in (54.0, 64.0, 84.0, 94.0):
        add(f"b1_yaw1_{value:+04.0f}", site_yaw_deg=(-43.0, value))
    for value in (-18.0, -13.0, -3.0, 2.0):
        add(f"b1_tilt0_{value:+04.0f}", site_tilt_deg=(value, -8.0))
    for value in (-18.0, -13.0, -3.0, 2.0):
        add(f"b1_tilt1_{value:+04.0f}", site_tilt_deg=(-8.0, value))
    if len(output) != 17 or len({config_signature(value) for value in output}) != 17:
        raise RuntimeError("E201 batch1 must contain 17 unique configurations")
    return output


def batch2_configs() -> list[SearchConfig]:
    base = replace(baseline(), name="b2_base", family="direction_batch2")
    output = [base]

    def add(name: str, **changes: Any) -> None:
        output.append(replace(base, name=name, **changes))

    for value in (-48.0, -45.5, -40.5, -38.0):
        add(f"b2_yaw0_{value:+05.1f}", site_yaw_deg=(value, 74.0))
    for value in (69.0, 71.5, 76.5, 79.0):
        add(f"b2_yaw1_{value:+05.1f}", site_yaw_deg=(-43.0, value))
    for value in (-10.5, -9.25, -6.75, -5.5):
        add(f"b2_tilt0_{value:+05.2f}", site_tilt_deg=(value, -8.0))
    for value in (-10.5, -9.25, -6.75, -5.5):
        add(f"b2_tilt1_{value:+05.2f}", site_tilt_deg=(-8.0, value))
    if len(output) != 17 or len({config_signature(value) for value in output}) != 17:
        raise RuntimeError("E201 batch2 must contain 17 unique configurations")
    return output


def batch3_configs() -> list[SearchConfig]:
    base = replace(
        baseline(),
        name="b3_base",
        family="array_batch3",
        site_tilt_deg=(-6.75, -8.0),
    )
    output = [base]

    def add(name: str, **changes: Any) -> None:
        output.append(replace(base, name=name, **changes))

    for value in (2.0, 3.0, 5.0, 6.0):
        add(f"b3_vspace_{value:.1f}", tx_vertical_spacing=value)
    for value in (0.4, 0.6):
        add(f"b3_hspace_{value:.1f}", tx_horizontal_spacing=value)
    add("b3_rx_fixed", rx_orientation="fixed")
    add("b3_physical_array", synthetic_array=False)
    if len(output) != 9 or len({config_signature(value) for value in output}) != 9:
        raise RuntimeError("E201 batch3 must contain 9 unique configurations")
    return output


def batch4_configs() -> list[SearchConfig]:
    base = replace(
        baseline(),
        name="b4_depth_16",
        family="depth_batch4",
        site_tilt_deg=(-6.75, -8.0),
    )
    output = [base]
    for value in (8, 12, 20, 24):
        output.append(
            replace(base, name=f"b4_depth_{value}", max_depth=value)
        )
    if len(output) != 5 or len({config_signature(value) for value in output}) != 5:
        raise RuntimeError("E201 batch4 must contain 5 unique configurations")
    return output


def batch5_configs() -> list[SearchConfig]:
    base = replace(
        baseline(),
        name="b5_freq_6690",
        family="frequency_batch5",
        site_tilt_deg=(-6.75, -8.0),
    )
    output = [base]
    for label, value in (("6688", 6.688e9), ("6693", 6.693e9), ("6696", 6.696e9)):
        output.append(replace(base, name=f"b5_freq_{label}", frequency_hz=value))
    if len(output) != 4 or len({config_signature(value) for value in output}) != 4:
        raise RuntimeError("E201 batch5 must contain 4 unique configurations")
    return output


__all__ = [
    "SearchConfig",
    "baseline",
    "batch1_configs",
    "batch2_configs",
    "batch3_configs",
    "batch4_configs",
    "batch5_configs",
    "config_signature",
    "portable_payload",
]
