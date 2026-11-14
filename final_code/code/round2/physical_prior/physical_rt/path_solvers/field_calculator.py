"""Computes the paths channel coefficients and delays"""

import mitsuba as mi
import drjit as dr
from typing import Callable, Tuple, List

from scipy.constants import speed_of_light

from round2.physical_prior.physical_rt.utils import rotation_matrix, jones_vec_dot, \
    spectrum_to_matrix_4f
from round2.physical_prior.physical_rt.antenna_pattern import antenna_pattern_to_world_implicit
from round2.physical_prior.physical_rt.constants import InteractionType
from .paths_buffer import PathsBuffer


class FieldCalculator:
    r"""
    Computes the channel coefficients and delays corresponding to traced paths

    The computation of the electric field is performed as follows:

        1. The electric field is initialized for every source from the antenna
    patterns.
        2. The electric field is "transported" by replaying the traced paths.
    At every interaction with the scene, the transfer matrix for this
    interaction is computed by evaluating the corresponding radio material, and
    the transfer matrix is applied to the transported electric field to update
    it.
        3. Once paths replay is over, the paths complex-valued coefficients are
    computed by taking the dot product between the transported electric field
    and the target antenna patterns.

    Note that the electric field is transported in the world implicit frame.
    """

    def __init__(self):

        # Dr.Jit mode for running the loop that peforms the transportation of
        # the electric field.
        # Symbolic mode is the fastest mode but does not currently support
        # backpropagation of gradients
        self._loop_mode = "symbolic"

    # pylint: disable=line-too-long
    def __call__(self,
                 wavelength: mi.Float | float,
                 paths: PathsBuffer,
                 samples_per_src: int,
                 src_positions: mi.Point3f,
                 tgt_positions: mi.Point3f,
                 src_orientations: mi.Point3f,
                 tgt_orientations: mi.Point3f,
                 src_antenna_patterns: List[Callable[[mi.Float,mi.Float], Tuple[mi.Complex2f,mi.Complex2f]]],
                 tgt_antenna_patterns: List[Callable[[mi.Float,mi.Float], Tuple[mi.Complex2f,mi.Complex2f]]],
                 ):
        r"""
        Computes the channel coefficients and delays for the given paths

        :param wavelength: Wavelength [m]
        :param paths: Traced paths
        :param samples_per_src: Number of samples per source
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param src_orientations: Sources orientations specified through three angles [rad] corresponding to a 3D rotation as defined in :eq:`rotation`
        :param tgt_orientations: Targets orientations specified through three angles [rad] corresponding to a 3D rotation as defined in :eq:`rotation`
        :param src_antenna_patterns: Antenna pattern of the sources
        :param tgt_antenna_patterns: Antenna pattern of the targets

        :return: Paths buffer with channel coefficients and delays set
        """

        # Return immediately if there are no paths
        if paths.buffer_size == 0:
            paths.a = dr.zeros(mi.Complex2f, 0)
            paths.tau = dr.zeros(mi.Float, 0)
            return paths

        # Compute the channel impulse response
        with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, False):
            self._compute_cir(wavelength, paths, samples_per_src,
                              src_positions, tgt_positions, src_orientations,
                              tgt_orientations, src_antenna_patterns,
                              tgt_antenna_patterns)

        return paths

    @property
    def loop_mode(self):
        r"""Get/set the Dr.Jit mode used to evaluate the loops that implement
        the solver. Should be one of "evaluated" or "symbolic". Symbolic mode
        (default) is the fastest one but does not support automatic
        differentiation.

        :type: "evaluated" | "symbolic"
        """
        return self._loop_mode

    @loop_mode.setter
    def loop_mode(self, mode):
        if mode not in ("evaluated", "symbolic"):
            raise ValueError("Invalid loop mode. Must be either 'evaluated'"
                             " or 'symbolic'")
        self._loop_mode = mode

    ##################################################
    # Internal methods
    ##################################################

    @dr.syntax
    def _compute_cir(
        self,
        wavelength: mi.Float | float,
        paths: PathsBuffer,
        samples_per_src: int,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f,
        src_orientations: mi.Point3f,
        tgt_orientations: mi.Point3f,
        src_antenna_patterns: List[Callable[[mi.Float,mi.Float],
                                   Tuple[mi.Complex2f,mi.Complex2f]]],
        tgt_antenna_patterns: List[Callable[[mi.Float,mi.Float],
                                   Tuple[mi.Complex2f,mi.Complex2f]]]
    ):
        """
        Computes the channel coefficients ``a`` and delays ``tau``.

        The paths buffer ``paths`` is updated in-place.

        :param wavelength: Wavelength [m]
        :param paths: Paths buffer. Updated in-place.
        :param samples_per_src: Number of samples per source
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param src_orientations: Sources orientations specified through three
                                 angles [rad] corresponding to a 3D rotation as
                                 defined in :eq:`rotation`.
        :param tgt_orientations: Targets orientations specified through three
                                 angles [rad] corresponding to a 3D rotation as
                                 defined in :eq:`rotation`
        :param src_antenna_patterns: Antenna pattern of the sources
        :param tgt_antenna_patterns: Antenna pattern of the targets
        """
        num_paths = paths.buffer_size
        max_depth = paths.max_depth

        # Source and target position for each path
        path_src_pos = dr.gather(mi.Point3f, src_positions,
                                 paths.source_indices)
        path_tgt_pos = dr.gather(mi.Point3f, tgt_positions,
                                 paths.target_indices)

        # Orientation of sources and targets and corresponding to-world
        # transforms
        path_src_ort = dr.gather(mi.Point3f, src_orientations,
                                 paths.source_indices)
        src_to_world = rotation_matrix(path_src_ort)
        path_tgt_ort = dr.gather(mi.Point3f, tgt_orientations,
                                 paths.target_indices)
        tgt_to_world = rotation_matrix(path_tgt_ort)

        # Current depth
        depth = dr.full(mi.UInt, 1, num_paths)

        # Interaction type
        interaction_type = paths.get_interaction_type(1, True)

        # Mask indicating which paths are active
        active = ( (depth <= max_depth) &
                   (interaction_type != InteractionType.NONE) &
                   paths.valid )

        # The following loop keeps track of the previous, current, and next path
        # vertices to compute the incident and scattered wave direction of
        # propagation and evaluate the radio material.

        # Previous vertex is initialized to source position
        prev_vertex = path_src_pos
        # Current vertex is initialized to the first one.
        # ~active corresponds to LoS paths.
        vertex = dr.select(~active, path_tgt_pos, paths.get_vertex(1, True))
        # Next vertex is initialized to 0
        next_vertex = dr.zeros(mi.Point3f, num_paths)

        # Initialize the incident direction of propagation of the wave
        # and the path length
        ki_world = mi.Vector3f(vertex - prev_vertex)
        path_length = dr.norm(ki_world)
        ki_world *= dr.rcp(path_length)

        # Direction of the scattered wave is initialized to 0
        ko_world = dr.zeros(mi.Vector3f, num_paths)

        # The following loop transports and update the electric field.
        # The electric field is initialized by the source antenna pattern
        e_fields = [antenna_pattern_to_world_implicit(src_antenna_pattern,
                                                      src_to_world, ki_world,
                                                      direction="out")
                    for src_antenna_pattern in src_antenna_patterns]

        # Solid angle of the ray tube.
        # It is required to compute the diffusely reflected field.
        # Initialized assuming that all the rays initially spawn from the source
        # share the unit sphere equally, i.e., initialized to
        # 4*PI/samples_per_src.
        # This quantity is also used to account for the fact that paths
        # including a diffuse reflection are sampled during shoot-and-bounce
        # with a probability defined by the radio material, and that this
        # probability should be canceled to avoid an undesired weighting that
        # arises from this sampling.
        # This canceling is implemented by scaling the solid-angle by the
        # inverse square-root of the probability of sampling the path.
        solid_angle = dr.full(mi.Float, 4.*dr.pi*dr.rcp(samples_per_src),
                              num_paths)

        # Length of the ray tube.
        # Note that this is different from the length of the path, as every
        # diffuse interaction generates a new ray tube, and therefore
        # effectively "resets" the ray tube length.
        ray_tube_length = dr.copy(path_length)

        # Exclude the non-loop variable `wavelength` to avoid having to
        # trace the loop twice, which is expensive.
        while dr.hint(active, mode=self.loop_mode, exclude=[wavelength]):

            # Flag set to True if this is the last depth
            last_depth = depth == max_depth

            # Flag indicating if data about the next depth can be gathered
            gather_next = active & ~last_depth

            # Mask indicating if this interaction is a diffuse reflection
            diffuse = active & (interaction_type == InteractionType.DIFFUSE)

            # Next interaction type
            next_interaction_type = paths.get_interaction_type(depth+1,
                                                               gather_next)
            # If the next interaction type is None, then this is the last
            # interaction
            next_is_none = next_interaction_type == InteractionType.NONE

            # Flag indicating if this interaction is the last one
            last_interaction = active & (next_is_none | last_depth)

            # Next vertex
            # Set to the target position is this is the last interaction
            next_vertex = dr.select(last_interaction,
                                    path_tgt_pos,
                                    paths.get_vertex(depth+1, gather_next))

            # Direction of the scattered wave.
            # Only updated if the path is still active, as we need a valid
            # value after the loop ends to evaluate the receive pattern
            ko_world = dr.select(active,
                                 mi.Vector3f(next_vertex - vertex),
                                 ko_world)
            length = dr.norm(ko_world)
            ko_world *= dr.rcp(length)

            # Intersected shape
            shape = paths.get_shape(depth, active)
            shape = dr.reinterpret_array(mi.ShapePtr, shape)

            # Update the fields
            # The solid angle of the ray tube is also updated based on the
            # probability of the interaction event
            e_fields, solid_angle = self._update_field(
                shape, paths, depth, interaction_type, ki_world, ko_world,
                e_fields, solid_angle, active)

            # Update the path length
            path_length += dr.select(active, length, 0.)

            # Updates the ray tube length
            ray_tube_length = dr.select(diffuse, 0.0, ray_tube_length)
            ray_tube_length += dr.select(active, length, 0.)

            # Update the solid angle
            # If a diffuse reflection is sampled, then it is set to 2PI.
            # Otherwise it is left unchanged
            solid_angle = dr.select(diffuse, dr.two_pi, solid_angle)

            # Prepare for next iteration
            depth += 1
            ki_world = dr.select(active, ko_world, ki_world)
            active &= (depth <= max_depth) & ~next_is_none
            prev_vertex = dr.copy(vertex)
            vertex = dr.copy(next_vertex)
            interaction_type = dr.copy(next_interaction_type)

        # Scaling to apply free-space propagation loss
        spreading_factor = dr.rcp(ray_tube_length)

        # Scaling by wavelength
        wl_scaling = wavelength * dr.rcp(4.*dr.pi)

        # Receive antenna pattern
        tgt_patterns = [antenna_pattern_to_world_implicit(tgt_antenna_pattern,
                                                          tgt_to_world,
                                                          -ki_world,
                                                          direction="in")
                        for tgt_antenna_pattern in tgt_antenna_patterns]

        # Compute channel coefficients
        # a[n][m] corresponds to the channel coefficient for the n^th receiver
        # antenna pattern and m^th transmitter antenna pattern
        a = []
        valid_a = mi.Bool(False)
        for tgt_pattern in tgt_patterns:
            a.append([])
            for e_field in e_fields:
                a_ = jones_vec_dot(tgt_pattern, e_field)
                a_ *= spreading_factor*wl_scaling
                valid_a |= (dr.abs(a_) > 0.)
                a[-1].append(a_)

        # Disable paths with 0 contribution
        paths.valid &= valid_a

        # Delay
        tau = path_length*dr.rcp(speed_of_light)

        paths.a = a
        paths.tau = tau

    def _update_field(self,
                      shape: mi.ShapePtr,
                      paths: PathsBuffer,
                      depth: mi.UInt,
                      interaction_type: mi.UInt,
                      ki_world: mi.Vector3f,
                      ko_world: mi.Vector3f,
                      e_fields: List[mi.Vector4f],
                      solid_angle: mi.Float,
                      active: mi.Bool) -> Tuple[mi.Matrix4f, mi.Float]:
        # pylint: disable=line-too-long
        r"""
        Evaluates the radio material and updates the electric field accordingly

        :param shape: Intersected shape
        :param paths: Buffer containing path information
        :param depth: Current depth/interaction index in the path
        :param interaction_type: Interaction type to evaluate, represented using :data:`~round2.physical_prior.physical_rt.constants.InteractionType`
        :param ki_world: Directions of propagation of the incident waves in the world frame
        :param ko_world: Directions of propagation of the scattered waves in the world frame
        :param e_fields: Jones vector representing the electric field as a 4D real-valued vector
        :param solid_angle: Ray tube solid angles [sr]
        :param active: Mask to specify active rays

        :return: Updated electric field and updated ray tube solid angle [sr]
        """

        # Radio material of the intersected shape
        rm = shape.bsdf()

        num_paths = paths.buffer_size

        # Normal to the intersected surface in the world frame
        normal_world = paths.get_primitive_props(depth, return_normal=True,
                                                 return_vertices=False, active=active)

        # Build a surface interaction object and context object to call the
        # radio material
        si = dr.zeros(mi.SurfaceInteraction3f, num_paths)
        ctx = mi.BSDFContext(mode=mi.TransportMode.Importance,
                             type_mask=0, component=0)

        # Ensure the normal is oriented in the opposite of the direction of
        # propagation of the incident wave
        normal_world *= dr.sign(dr.dot(normal_world, -ki_world))
        si.n = normal_world
        si.sh_frame.n = normal_world
        si.initialize_sh_frame()

        # Set `si.wi` to the local direction of propagation of the incident wave
        si.wi = si.to_local(ki_world)
        # Interaction point
        si.p = paths.get_vertex(depth, active)
        # Intersected shape
        si.shape = shape
        # Intersected primitive
        si.prim_index = paths.get_primitive(depth, active)

        # The final component stores the interaction type for material replay.
        si.dp_du = mi.Vector3f(0.,
                               0.,
                               dr.reinterpret_array(mi.Float,
                                                    interaction_type))

        # Probability of the event to be sampled
        probs = paths.get_prob(depth, active)
        # Scale the solid angle accordingly
        solid_angle[active] *= dr.rcp(probs)

        # Update the fields
        for i, e_field in enumerate(e_fields):
            # `si.duv_dx` and `si.duv_dy` stores the incident field
            si.duv_dx = mi.Vector2f(e_field.x, # S
                                    e_field.y) # P
            si.duv_dy = mi.Vector2f(e_field.z, # S
                                    e_field.w) # P
            # `si.t` stores the solid angle
            si.t = solid_angle

            # Evaluate the radio material
            jones_mat = rm.eval(ctx, si, ko_world, active)
            jones_mat = spectrum_to_matrix_4f(jones_mat)
            # Update the field by applying the Jones matrix
            e_fields[i] = dr.select(active, jones_mat@e_field, e_field)

        return e_fields, solid_angle
