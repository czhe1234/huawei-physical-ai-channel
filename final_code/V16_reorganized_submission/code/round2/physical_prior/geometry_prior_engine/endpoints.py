import drjit as dr
import mitsuba as mi
from typing_extensions import Self

from .utils import theta_phi_from_unit_vec


class SceneEndpoint:
    def __init__(
        self,
        name: str,
        position: mi.Point3f,
        look_at: mi.Point3f | Self,
    ) -> None:
        self.name = name
        self.position = mi.Point3f(position)
        self.look_at(look_at)

    def look_at(self, target: mi.Point3f | Self) -> None:
        target_position = (
            target.position if isinstance(target, SceneEndpoint) else target
        )
        direction = dr.normalize(mi.Point3f(target_position) - self.position)
        theta, phi = theta_phi_from_unit_vec(direction)
        self.orientation = mi.Point3f(phi, theta - dr.pi / 2.0, 0.0)


class Source(SceneEndpoint):
    pass


class Target(SceneEndpoint):
    pass


__all__ = ["SceneEndpoint", "Source", "Target"]
