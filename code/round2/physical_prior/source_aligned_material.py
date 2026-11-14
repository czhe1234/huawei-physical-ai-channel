"""ITU concrete material tables for the source-aligned propagation solver.

The equations follow ITU-R P.2040 single-layer slab coefficients, matching the
public Sionna 2.0.1 propagation semantics used to generate the E185 reference.
The implementation is independent NumPy code and has no Sionna dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON_0_F_M = 8.854_187_812_8e-12
LIGHT_SPEED_M_S = 299_792_458.0


@dataclass(frozen=True)
class MaterialTable:
    cosine: np.ndarray
    specular_probability: np.ndarray
    diffuse_probability: np.ndarray
    transmission_probability: np.ndarray
    reflection_power: np.ndarray
    transmission_power: np.ndarray


def itu_concrete_properties(frequency_hz: float) -> tuple[float, float]:
    """Return ITU-R P.2040 concrete relative permittivity and conductivity."""

    frequency_ghz = float(frequency_hz) / 1e9
    relative_permittivity = 5.24
    conductivity = 0.0462 * frequency_ghz**0.7822
    return relative_permittivity, conductivity


def single_layer_slab_coefficients(
    cosine: np.ndarray,
    *,
    frequency_hz: float,
    thickness_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute complex TE/TM reflection and transmission coefficients."""

    cosine = np.asarray(cosine, dtype=np.float64)
    relative_permittivity, conductivity = itu_concrete_properties(frequency_hz)
    angular_frequency = 2.0 * np.pi * float(frequency_hz)
    wavelength = LIGHT_SPEED_M_S / float(frequency_hz)
    eta = relative_permittivity - 1j * conductivity / (
        angular_frequency * EPSILON_0_F_M
    )
    sin_theta_squared = 1.0 - np.square(cosine)
    root = np.sqrt(eta - sin_theta_squared + 0j)

    r_te_prime = (cosine - root) / (cosine + root)
    r_tm_prime = (eta * cosine - root) / (eta * cosine + root)
    q = 2.0 * np.pi * float(thickness_m) / wavelength * root
    exp_minus_jq = np.exp(-1j * q)
    exp_minus_j2q = np.exp(-2j * q)

    r_te_squared = np.square(r_te_prime)
    r_tm_squared = np.square(r_tm_prime)
    denominator_te = 1.0 - r_te_squared * exp_minus_j2q
    denominator_tm = 1.0 - r_tm_squared * exp_minus_j2q
    r_te = r_te_prime * (1.0 - exp_minus_j2q) / denominator_te
    r_tm = r_tm_prime * (1.0 - exp_minus_j2q) / denominator_tm
    t_te = (1.0 - r_te_squared) * exp_minus_jq / denominator_te
    t_tm = (1.0 - r_tm_squared) * exp_minus_jq / denominator_tm
    return r_te, r_tm, t_te, t_tm


def build_material_table(
    *,
    frequency_hz: float,
    thickness_m: float,
    scattering_coefficient: float,
    samples: int = 16_385,
) -> MaterialTable:
    """Build an interpolation table for event sampling and scalar field power."""

    if samples < 2:
        raise ValueError("material table requires at least two samples")
    scattering_coefficient = float(scattering_coefficient)
    if not 0.0 <= scattering_coefficient <= 1.0:
        raise ValueError("scattering coefficient must be in [0, 1]")

    cosine = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    r_te, r_tm, t_te, t_tm = single_layer_slab_coefficients(
        cosine,
        frequency_hz=frequency_hz,
        thickness_m=thickness_m,
    )
    reflection = np.square(np.abs(r_te)) + np.square(np.abs(r_tm))
    transmission = np.square(np.abs(t_te)) + np.square(np.abs(t_tm))
    scatter_power = scattering_coefficient**2
    specular_weight = reflection * (1.0 - scatter_power)
    diffuse_weight = reflection * scatter_power
    transmission_weight = transmission
    total = specular_weight + diffuse_weight + transmission_weight
    valid = total > 0.0

    specular_probability = np.divide(
        specular_weight,
        total,
        out=np.zeros_like(total),
        where=valid,
    )
    diffuse_probability = np.divide(
        diffuse_weight,
        total,
        out=np.zeros_like(total),
        where=valid,
    )
    transmission_probability = np.divide(
        transmission_weight,
        total,
        out=np.zeros_like(total),
        where=valid,
    )

    # The PathToken only retains normalized relative energy. Averaging the two
    # orthogonal incident polarizations preserves the source solver's material
    # power ordering without importing a final Sionna channel coefficient.
    reflection_power = 0.5 * reflection
    transmission_power = 0.5 * transmission
    arrays = {
        "cosine": cosine,
        "specular_probability": specular_probability,
        "diffuse_probability": diffuse_probability,
        "transmission_probability": transmission_probability,
        "reflection_power": reflection_power,
        "transmission_power": transmission_power,
    }
    return MaterialTable(
        **{
            name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in arrays.items()
        }
    )

