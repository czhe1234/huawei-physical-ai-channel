"""Diffuse-scattering pattern used by the production solver."""

from abc import ABC, abstractmethod

import drjit as dr
import mitsuba as mi


class ScatteringPattern(ABC):
    """Interface implemented by material scattering patterns."""

    @abstractmethod
    def __call__(
        self,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
    ) -> mi.Float:
        """Evaluate the normalized scattering density."""


class LambertianPattern(ScatteringPattern):
    """Lambertian diffuse-scattering density."""

    def __call__(
        self,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
    ) -> mi.Float:
        del ki_local
        return dr.abs(ko_local.z) / dr.pi


def create_scattering_pattern(
    pattern: str | ScatteringPattern | None,
    **attributes,
) -> ScatteringPattern:
    """Build the only scattering model enabled by production configs."""

    if isinstance(pattern, ScatteringPattern):
        if attributes:
            raise ValueError(
                "scattering-pattern attributes require a named factory"
            )
        return pattern
    if pattern not in (None, "lambertian"):
        raise ValueError(
            "only the production scattering pattern 'lambertian' is supported"
        )
    if attributes:
        names = ", ".join(sorted(attributes))
        raise ValueError(f"unsupported Lambertian attributes: {names}")
    return LambertianPattern()


__all__ = [
    "LambertianPattern",
    "ScatteringPattern",
    "create_scattering_pattern",
]
