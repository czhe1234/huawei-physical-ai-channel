"""Geometry-conditioned propagation engine for offline prior construction."""

import mitsuba as mi

if mi.variant() is None:
    try:
        mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
    except ImportError:
        mi.set_variant("llvm_ad_mono_polarized")

from .antenna_array import PlanarArray
from .radio_devices import Receiver, Transmitter
from .scene import load_scene
from .propagation_support import PropagationSupportBuilder

__version__ = "1.0.0"

__all__ = [
    "PropagationSupportBuilder",
    "PlanarArray",
    "Receiver",
    "Transmitter",
    "load_scene",
]
