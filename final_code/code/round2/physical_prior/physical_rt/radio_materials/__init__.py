"""Radio-material implementation required by the propagation solver."""

from .radio_material_base import RadioMaterialBase
from .radio_material import RadioMaterial
from .itu_material import ITURadioMaterial
from .scattering_pattern import LambertianPattern, ScatteringPattern

__all__ = [
    "ITURadioMaterial",
    "LambertianPattern",
    "RadioMaterial",
    "RadioMaterialBase",
    "ScatteringPattern",
]
