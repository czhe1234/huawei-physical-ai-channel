"""Internal helpers shared by the compact propagation runtime."""

from .electromagnetics import (
    complex_relative_permittivity,
    itu_coefficients_single_layer_slab,
)
from .geometry import (
    point_plane_reflection,
    rotation_matrix,
    theta_phi_from_unit_vec,
)
from .hashing import PlaneHasher, hash_fnv1a
from .jones import (
    jones_matrix_rotator_flip_forward,
    jones_matrix_to_world_implicit,
    jones_vec_dot,
    to_world_jones_rotator,
)
from .misc import spectrum_to_matrix_4f
from .scene_intersections import (
    fibonacci_lattice,
    offset_p,
    spawn_ray_from_sources,
    spawn_ray_to,
)

__all__ = [
    "PlaneHasher",
    "complex_relative_permittivity",
    "fibonacci_lattice",
    "hash_fnv1a",
    "itu_coefficients_single_layer_slab",
    "jones_matrix_rotator_flip_forward",
    "jones_matrix_to_world_implicit",
    "jones_vec_dot",
    "offset_p",
    "point_plane_reflection",
    "rotation_matrix",
    "spawn_ray_from_sources",
    "spawn_ray_to",
    "spectrum_to_matrix_4f",
    "theta_phi_from_unit_vec",
    "to_world_jones_rotator",
]
