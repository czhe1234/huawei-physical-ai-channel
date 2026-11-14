"""Source-aligned physical propagation runtime.

The propagation kernels are derived from the Sionna RT 2.0.1 implementation
used by the production physical solver. This package does not import the
``sionna`` package.
"""

import mitsuba as mi

if mi.variant() is None:
    try:
        mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
    except ImportError:
        mi.set_variant("llvm_ad_mono_polarized")

from .utils import *
from .antenna_pattern import (
    AntennaPattern,
    PolarizedAntennaPattern,
    register_antenna_pattern,
    register_polarization,
    register_polarization_model,
)
from .antenna_array import AntennaArray, PlanarArray
from .radio_devices import RadioDevice, Receiver, Transmitter
from .scene import Scene, load_scene, load_scene_from_string
from .radio_materials import *
from .constants import InteractionType
from .path_solvers import PathSolver, Paths

__version__ = "2.0.1-e185-port"
