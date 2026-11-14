"""Radio-material implementation required by the propagation solver."""

from .radio_material_base import RadioMaterialBase
from .radio_material import RadioMaterial
from .itu_material import ITURadioMaterial
from .scattering_pattern import (
    BackscatteringPattern,
    DirectivePattern,
    LambertianPattern,
    ScatteringPattern,
    register_scattering_pattern,
)

__all__ = [
    "BackscatteringPattern",
    "DirectivePattern",
    "ITURadioMaterial",
    "LambertianPattern",
    "RadioMaterial",
    "RadioMaterialBase",
    "ScatteringPattern",
    "register_scattering_pattern",
]
