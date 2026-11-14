"""Antenna-array geometry used by the production propagation solver."""

import drjit as dr
import mitsuba as mi

from .antenna_pattern import AntennaPattern, create_antenna_pattern
from .utils import rotation_matrix


class AntennaArray:
    """An antenna pattern placed at wavelength-normalized positions."""

    def __init__(
        self,
        antenna_pattern: AntennaPattern,
        normalized_positions: mi.Point3f,
    ) -> None:
        self.antenna_pattern = antenna_pattern
        self.normalized_positions = normalized_positions

    @property
    def antenna_pattern(self) -> AntennaPattern:
        return self._antenna_pattern

    @antenna_pattern.setter
    def antenna_pattern(self, value: AntennaPattern) -> None:
        if not isinstance(value, AntennaPattern):
            raise TypeError("antenna_pattern must be an AntennaPattern")
        self._antenna_pattern = value

    @property
    def normalized_positions(self) -> mi.Point3f:
        return self._normalized_positions

    @normalized_positions.setter
    def normalized_positions(self, value) -> None:
        self._normalized_positions = mi.Point3f(value)

    @property
    def num_ant(self) -> int:
        return (
            dr.shape(self._normalized_positions)[1]
            * len(self.antenna_pattern.patterns)
        )

    @property
    def array_size(self) -> int:
        return dr.shape(self._normalized_positions)[1]

    def positions(self, wavelength: float) -> mi.Point3f:
        return self._normalized_positions * wavelength

    def rotate(self, wavelength: float, orientation: mi.Point3f) -> mi.Point3f:
        return rotation_matrix(orientation) @ self.positions(wavelength)


class PlanarArray(AntennaArray):
    """Regular column-first array in the local y-z plane."""

    def __init__(
        self,
        *,
        num_rows: int,
        num_cols: int,
        vertical_spacing: float = 0.5,
        horizontal_spacing: float = 0.5,
        pattern: str,
        **pattern_kwargs,
    ) -> None:
        antenna_pattern = create_antenna_pattern(pattern, **pattern_kwargs)
        normalized_positions = dr.zeros(mi.Point3f, num_rows * num_cols)
        row, col = dr.meshgrid(
            dr.arange(mi.UInt, num_rows),
            dr.arange(mi.UInt, num_cols),
        )
        normalized_positions.y = (
            horizontal_spacing * col
            - (num_cols - 1) * horizontal_spacing / 2
        )
        normalized_positions.z = (
            -vertical_spacing * row
            + (num_rows - 1) * vertical_spacing / 2
        )
        super().__init__(antenna_pattern, normalized_positions)


__all__ = ["AntennaArray", "PlanarArray"]
