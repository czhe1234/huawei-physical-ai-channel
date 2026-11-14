"""Compact radio-propagation runtime used by the Round 2 pipeline."""

import mitsuba as mi

if mi.variant() is None:
    try:
        mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
    except ImportError:
        mi.set_variant("llvm_ad_mono_polarized")

from .antenna_array import PlanarArray
from .radio_devices import Receiver, Transmitter
from .scene import load_scene
from .path_solvers import PathSolver

__version__ = "1.0.0"

__all__ = [
    "PathSolver",
    "PlanarArray",
    "Receiver",
    "Transmitter",
    "load_scene",
]
