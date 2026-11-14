"""Radio endpoint position and orientation."""

import drjit as dr
import mitsuba as mi
from typing_extensions import Self

from ..utils import theta_phi_from_unit_vec


class RadioDevice:
    """A named radio endpoint with a position and orientation."""

    def __init__(
        self,
        name: str,
        position: mi.Point3f,
        orientation: mi.Point3f | None = None,
        look_at: mi.Point3f | Self | None = None,
    ) -> None:
        self._name = name
        self.position = position
        if orientation is not None and look_at is not None:
            raise ValueError("only one of orientation or look_at can be specified")
        if look_at is None:
            self.orientation = mi.Point3f(0) if orientation is None else orientation
        else:
            self.look_at(look_at)

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> mi.Point3f:
        return self._position

    @position.setter
    def position(self, value) -> None:
        self._position = mi.Point3f(value)

    @property
    def orientation(self) -> mi.Point3f:
        return self._orientation

    @orientation.setter
    def orientation(self, value) -> None:
        self._orientation = mi.Point3f(value)

    def look_at(self, target: mi.Point3f | Self) -> None:
        """Orient the local x-axis toward a position or another endpoint."""

        target_position = target.position if isinstance(target, RadioDevice) else target
        direction = dr.normalize(mi.Point3f(target_position) - self.position)
        theta, phi = theta_phi_from_unit_vec(direction)
        self.orientation = mi.Point3f(phi, theta - dr.pi / 2.0, 0.0)
