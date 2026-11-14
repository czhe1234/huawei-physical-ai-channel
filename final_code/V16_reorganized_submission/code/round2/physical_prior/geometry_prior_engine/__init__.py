import mitsuba as mi

if mi.variant() is None:
    try:
        mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
    except ImportError:
        mi.set_variant("llvm_ad_mono_polarized")

from .array_model import build_receive_array, build_transmit_array
from .endpoints import Source, Target
from .scene import load_scene
from .scene_components import ScenePriorBuilder

__version__ = "1.0.0"

__all__ = [
    "ScenePriorBuilder",
    "build_receive_array",
    "build_transmit_array",
    "Source",
    "Target",
    "load_scene",
]
