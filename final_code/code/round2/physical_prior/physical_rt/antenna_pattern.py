"""Antenna patterns used by the production propagation solver."""

from collections.abc import Callable

import drjit as dr
import mitsuba as mi

from .utils import (
    jones_matrix_rotator_flip_forward,
    theta_phi_from_unit_vec,
    to_world_jones_rotator,
)


_POLARIZATION_SLANTS = {
    "V": (0.0,),
    "H": (dr.pi / 2,),
    "VH": (0.0, dr.pi / 2),
    "cross": (-dr.pi / 4, dr.pi / 4),
}


def _v_dipole_pattern(theta: mi.Float, phi: mi.Float) -> mi.Complex2f:
    """Vertically polarized short-dipole pattern."""

    del phi
    c_theta = dr.abs(dr.sqrt(1.5) * dr.sin(theta))
    return mi.Complex2f(c_theta, 0)


def _v_tr38901_pattern(theta: mi.Float, phi: mi.Float) -> mi.Complex2f:
    """Vertically polarized 3GPP TR 38.901 element pattern."""

    phi = phi + dr.pi
    phi -= dr.floor(phi / (2.0 * dr.pi)) * 2.0 * dr.pi
    phi -= dr.pi

    theta_3db = phi_3db = 65.0 / 180.0 * dr.pi
    a_max = sla_v = 30.0
    g_e_max = 8.0
    a_v = -dr.min([12.0 * ((theta - dr.pi / 2.0) / theta_3db) ** 2, sla_v])
    a_h = -dr.min([12.0 * (phi / phi_3db) ** 2, a_max])
    a_db = -dr.min([-(a_v + a_h), a_max]) + g_e_max
    c_theta = dr.sqrt(dr.power(10.0, a_db / 10.0))
    return mi.Complex2f(c_theta, 0)


_VERTICAL_PATTERNS = {
    "dipole": _v_dipole_pattern,
    "tr38901": _v_tr38901_pattern,
}


class AntennaPattern:
    """One or two callable polarization patterns."""

    def __init__(self, patterns: list[Callable] | None = None) -> None:
        self.patterns = [] if patterns is None else patterns

    def __getitem__(self, index: int):
        return self.patterns[index]

    @property
    def patterns(self) -> list[Callable]:
        return self._patterns

    @patterns.setter
    def patterns(self, value: list[Callable]) -> None:
        if not isinstance(value, list) or len(value) > 2:
            raise ValueError("patterns must be a list of length at most two")
        self._patterns = value


class PolarizedAntennaPattern(AntennaPattern):
    """Apply the TR 38.901 model-2 polarization transform."""

    def __init__(
        self,
        vertical_pattern: Callable,
        polarization: str,
        polarization_model: str = "tr38901_2",
    ) -> None:
        if polarization_model != "tr38901_2":
            raise ValueError(
                "only the production polarization model 'tr38901_2' is supported"
            )
        try:
            slant_angles = _POLARIZATION_SLANTS[polarization]
        except KeyError as exc:
            supported = ", ".join(sorted(_POLARIZATION_SLANTS))
            raise ValueError(
                f"unsupported polarization '{polarization}'; choose {supported}"
            ) from exc

        patterns = []
        for slant_angle in slant_angles:

            def pattern(theta, phi, angle=slant_angle):
                c_theta_tilde = vertical_pattern(theta, phi)
                sin_angle, cos_angle = dr.sincos(angle)
                return cos_angle * c_theta_tilde, sin_angle * c_theta_tilde

            patterns.append(pattern)
        super().__init__(patterns)


def create_antenna_pattern(
    name: str,
    *,
    polarization: str,
    polarization_model: str = "tr38901_2",
) -> AntennaPattern:
    """Build one of the two antenna patterns used by production."""

    try:
        vertical_pattern = _VERTICAL_PATTERNS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_VERTICAL_PATTERNS))
        raise ValueError(
            f"unsupported antenna pattern '{name}'; choose {supported}"
        ) from exc
    return PolarizedAntennaPattern(
        vertical_pattern,
        polarization,
        polarization_model,
    )


def antenna_pattern_to_world_implicit(
    pattern: Callable,
    to_world: mi.Matrix3f,
    k_world: mi.Vector3f,
    direction: str,
) -> mi.Vector4f:
    """Evaluate an antenna pattern in the world implicit Jones basis."""

    to_local = to_world.T
    k_local = to_local @ k_world
    theta_local, phi_local = theta_phi_from_unit_vec(k_local)
    c_theta, c_phi = pattern(theta_local, phi_local)
    f_real = mi.Vector2f(c_theta.real, c_phi.real)
    f_imag = mi.Vector2f(c_theta.imag, c_phi.imag)

    rotator = to_world_jones_rotator(to_world, k_local)
    if direction == "in":
        rotator = jones_matrix_rotator_flip_forward(k_world) @ rotator

    f_real = rotator @ f_real
    f_imag = rotator @ f_imag
    return mi.Vector4f(f_real.x, f_real.y, f_imag.x, f_imag.y)


__all__ = [
    "AntennaPattern",
    "PolarizedAntennaPattern",
    "antenna_pattern_to_world_implicit",
    "create_antenna_pattern",
]
