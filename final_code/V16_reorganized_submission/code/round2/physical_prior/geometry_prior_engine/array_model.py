from collections.abc import Callable

import drjit as dr
import mitsuba as mi

from .utils import (
    jones_matrix_rotator_flip_forward,
    rotation_matrix,
    theta_phi_from_unit_vec,
    to_world_jones_rotator,
)


def _receive_element(theta: mi.Float, phi: mi.Float) -> mi.Complex2f:
    del phi
    return mi.Complex2f(dr.abs(dr.sqrt(1.5) * dr.sin(theta)), 0)


def _transmit_element(theta: mi.Float, phi: mi.Float) -> mi.Complex2f:
    phi = phi + dr.pi
    phi -= dr.floor(phi / (2.0 * dr.pi)) * 2.0 * dr.pi
    phi -= dr.pi
    beamwidth = 65.0 / 180.0 * dr.pi
    attenuation_limit = 30.0
    vertical = -dr.min(
        [12.0 * ((theta - dr.pi / 2.0) / beamwidth) ** 2, attenuation_limit]
    )
    horizontal = -dr.min(
        [12.0 * (phi / beamwidth) ** 2, attenuation_limit]
    )
    gain_db = -dr.min([-(vertical + horizontal), attenuation_limit]) + 8.0
    return mi.Complex2f(dr.sqrt(dr.power(10.0, gain_db / 10.0)), 0)


def _polarized_elements(
    vertical_response: Callable,
    slant_angles: tuple[float, float],
) -> list[Callable]:
    elements = []
    for slant_angle in slant_angles:

        def response(theta, phi, angle=slant_angle):
            vertical = vertical_response(theta, phi)
            sin_angle, cos_angle = dr.sincos(angle)
            return cos_angle * vertical, sin_angle * vertical

        elements.append(response)
    return elements


def _planar_positions(
    rows: int,
    columns: int,
    vertical_spacing: float,
    horizontal_spacing: float,
) -> mi.Point3f:
    positions = dr.zeros(mi.Point3f, rows * columns)
    row, column = dr.meshgrid(
        dr.arange(mi.UInt, rows),
        dr.arange(mi.UInt, columns),
    )
    positions.y = horizontal_spacing * column - (
        columns - 1
    ) * horizontal_spacing / 2
    positions.z = -vertical_spacing * row + (rows - 1) * vertical_spacing / 2
    return positions


class ArrayModel:
    def __init__(
        self,
        elements: list[Callable],
        normalized_positions: mi.Point3f,
    ) -> None:
        self.elements = elements
        self.normalized_positions = mi.Point3f(normalized_positions)

    @property
    def num_ant(self) -> int:
        return dr.shape(self.normalized_positions)[1] * len(self.elements)

    @property
    def array_size(self) -> int:
        return dr.shape(self.normalized_positions)[1]

    def rotate(self, wavelength: float, orientation: mi.Point3f) -> mi.Point3f:
        return rotation_matrix(orientation) @ (self.normalized_positions * wavelength)


def build_transmit_array(
    vertical_spacing: float,
    horizontal_spacing: float,
) -> ArrayModel:
    return ArrayModel(
        _polarized_elements(_transmit_element, (0.0, dr.pi / 2)),
        _planar_positions(8, 16, vertical_spacing, horizontal_spacing),
    )


def build_receive_array() -> ArrayModel:
    return ArrayModel(
        _polarized_elements(_receive_element, (-dr.pi / 4, dr.pi / 4)),
        _planar_positions(2, 1, 0.5, 0.5),
    )


def element_response_to_world(
    response: Callable,
    to_world: mi.Matrix3f,
    direction_world: mi.Vector3f,
    incoming: bool,
) -> mi.Vector4f:
    to_local = to_world.T
    direction_local = to_local @ direction_world
    theta_local, phi_local = theta_phi_from_unit_vec(direction_local)
    c_theta, c_phi = response(theta_local, phi_local)
    field_real = mi.Vector2f(c_theta.real, c_phi.real)
    field_imag = mi.Vector2f(c_theta.imag, c_phi.imag)
    rotator = to_world_jones_rotator(to_world, direction_local)
    if incoming:
        rotator = jones_matrix_rotator_flip_forward(direction_world) @ rotator
    field_real = rotator @ field_real
    field_imag = rotator @ field_imag
    return mi.Vector4f(field_real.x, field_real.y, field_imag.x, field_imag.y)


__all__ = [
    "ArrayModel",
    "build_receive_array",
    "build_transmit_array",
    "element_response_to_world",
]
