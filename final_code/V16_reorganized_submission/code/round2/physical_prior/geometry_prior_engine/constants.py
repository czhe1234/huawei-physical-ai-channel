import drjit as dr
import mitsuba as mi
import numpy as np


class InteractionType:

    NONE = 0
    SPECULAR = 1 << 0
    DIFFUSE = 1 << 1
    REFRACTION = 1 << 2


DEFAULT_FREQUENCY = 3.5e9
NO_JONES_MATRIX = 1 << 10
DEFAULT_THICKNESS = 0.1
INVALID_SHAPE = np.uint32(0xFFFFFFFF)
INVALID_PRIMITIVE = np.uint32(0xFFFFFFFF)
EPSILON_FLOAT = dr.epsilon(mi.Float) * 100.0
MIN_SEGMENT_LENGTH = 1e-2
