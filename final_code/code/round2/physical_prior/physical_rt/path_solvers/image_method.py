"""Implementation of the image method"""

import drjit as dr
import mitsuba as mi
from typing import Tuple

from round2.physical_prior.physical_rt.utils import (
    offset_p,
    point_plane_reflection,
    spawn_ray_to,
)
from round2.physical_prior.physical_rt.constants import InteractionType, MIN_SEGMENT_LENGTH
from .paths_buffer import PathsBuffer, PathsBufferBase


class ImageMethod:
    r"""
    Image method for evaluating specular chains and specular suffixes candidates

    A specular chain (suffix) is a path (suffix) that consists only of specular
    reflections and refractions.

    This class processes the candidate specular suffixes and specular chains
    using the image method to compute valid paths from the candidates specular
    chains and suffixes. It consists in 5 steps:

    1. The depth at which the specular suffix starts is determined.
    This depth corresponds to the lowest depth from which the path consists only
    of specular reflections and refraction. For specular chains, this depth
    equals to one.

    2. The source of the specular suffix is determined. For specular chains, it
    is simply the source of the path. For specular suffixes, it is
    the intersection point preceding the specular suffix.

    4. Image computations: Images of the sources are computed by reflecting them
    on surfaces on which a specular reflection occurred. Refraction events
    are ignored during this step.

    5. Backtracking: Tracing is done backward from the targets to the sources,
    by spawning rays toward the images, first from the target, then from the
    intersection points of the rays with the scene. These intersection
    points are the final vertices of the paths for specular reflections and
    refractions. If intersections with other primitives than the ones computed
    during the candidates generation are found, then the paths are discarded,
    i.e., flagged as a non-valid candidate.
    """

    def __init__(self):

        # Dr.Jit mode for running the loops that implement the image method
        # Symbolic mode is the fastest mode but does not currently support
        # backpropagation of gradients
        self._loop_mode = "symbolic"


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

    def __call__(self,
                 scene: mi.Scene,
                 paths: PathsBuffer,
                 src_positions: mi.Point3f,
                 tgt_positions: mi.Point3f) -> PathsBuffer:
        r"""
        Exectues the image method

        :param scene: Mitsuba scene
        :param paths: Candidate paths
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets

        :return: Processed paths
        """

        # Stop immediately if there are no paths
        if paths.buffer_size == 0:
            return paths

        # Candidates are the paths not marked as valid during the candidates
        # generation process
        valid_candidate = ~paths.valid

        # Gather the source and target of every path
        paths_sources, paths_targets\
            = self._gather_paths_sources_targets(paths, src_positions,
                                                 tgt_positions, valid_candidate)

        # Depth at which the specular suffixes start
        sf_start_depth = self._specular_chain_start_depth(paths,
                                                          valid_candidate)

        # Position of the specular suffixes sources
        sf_source = self._get_sf_source(paths, paths_sources, sf_start_depth,
                                        valid_candidate)

        ## Image method ##

        # Computes the images of the sources
        self._compute_images(paths, sf_source, sf_start_depth, valid_candidate)

        # Backtrack
        valid_candidate = self._backtrack(scene, paths, paths_targets,
                                          sf_source, sf_start_depth,
                                          valid_candidate)

        ## ## ## ## ## ##

        # Update the candidate valid status
        paths.valid |= valid_candidate
        dr.schedule(paths.valid)

        return paths

    ##################################################
    # Internal methods
    ##################################################

    def _gather_paths_sources_targets(self,
                                      paths,
                                      src_positions: mi.Point3f,
                                      tgt_positions: mi.Point3f,
                                      valid_candidate: mi.Bool
                                      ) -> Tuple[mi.Point3f, mi.Point3f]:
        r"""
        Gathers the sources and targets of every paths

        :param paths: Candidate paths
        :param src_positions: Positions of the sources
        :param tgt_positions  Positions of the targets
        :param valid_candidate: Flags specifying candidates marked as valid

        :return: Sources and targets of all paths
        """

        paths_sources = dr.gather(mi.Point3f, src_positions,
                                  paths.source_indices, valid_candidate)
        paths_targets = dr.gather(mi.Point3f, tgt_positions,
                                  paths.target_indices, valid_candidate)
        return paths_sources, paths_targets

    @dr.syntax
    def _specular_chain_start_depth(self,
                                    paths: PathsBuffer,
                                    valid_candidate: mi.Bool) -> mi.UInt:
        r"""
        Determines the depth at which the specular suffixes start

        :param paths: Candidate paths
        :param valid_candidate: Flags specifying candidates marked as valid

        :return: Depths at which the specular suffixes start
        """

        max_depth = paths.max_depth
        max_num_paths = paths.buffer_size

        # Array for storing the depth at which the specular suffix starts
        sf_start_depth = dr.full(mi.UInt, max_depth, max_num_paths)

        active = dr.copy(valid_candidate)
        depth = mi.UInt(max_depth)
        while dr.hint(active, mode=self.loop_mode):

            int_type = paths.get_interaction_type(depth, active)
            specular = int_type == InteractionType.SPECULAR
            refraction = int_type == InteractionType.REFRACTION
            none = int_type == InteractionType.NONE

            # If not in the specular suffix anymore, then deactivate the path to
            # stop updating the corresponding entry in `sf_start_depth`
            active &= (specular | refraction | none)

            # Update the starting point if the ray is active
            sf_start_depth = dr.select(active, depth, sf_start_depth)

            depth -= 1
            active &= depth > 0

        return sf_start_depth

    def _get_sf_source(self,
                       paths: PathsBuffer,
                       paths_sources: mi.Point3f,
                       sf_start_depth: mi.UInt,
                       valid_candidate: mi.Bool) -> mi.Point3f:
        r"""
        Returns the sources positions of the specular suffixes, which is
        * The source of the path if the path is a specular chain
        * The vertex of the last diffuse reflection if the path is not a
            specular chain but ends with a specular suffix

        :param paths: Candidate paths
        :param paths_sources: Sources of every paths
        :param sf_start_depth: Depths at which the specular suffixes start
        :param valid_candidate: Flags specifying candidates marked as valid

        :return: Positions of the sources of the specular suffixes
        """

        sf_source = sf_start_depth == 1

        last_diff_depth = sf_start_depth - 1
        last_diff_depth = dr.select(sf_source, 1, last_diff_depth)

        vertex = paths.get_vertex(last_diff_depth, valid_candidate)
        sf_source = dr.select(sf_source, paths_sources, vertex)

        return sf_source

    @dr.syntax
    def _compute_images(self,
                        paths: PathsBuffer,
                        sf_source: mi.Point3f,
                        sf_start_depth: mi.UInt,
                        valid_candidate: mi.Bool) -> None:
        r"""
        Computes the images of the sources

        This is the first step of the image method.
        The images coordinates are stored in ``path.vertices``.

        :param paths: Candidate paths
        :param sf_source: Positions of the sources of the specular suffixes
        :param sf_start_depth: Depths at which the specular suffixes starts
        :param valid_candidate: Flag specifying candidates marked as valid

        """

        max_depth = paths.max_depth
        num_paths = paths.buffer_size

        # Images of the source
        source_images = dr.copy(sf_source)

        normal = dr.zeros(mi.Vector3f, num_paths)
        active = dr.copy(valid_candidate)
        depth = dr.full(mi.UInt, 1, num_paths)
        # Temporary copy to avoid write-after-read in the symbolic loop.
        paths_base = PathsBufferBase.from_paths_buffer(paths)

        while dr.hint(active, mode=self.loop_mode):

            int_type = paths.get_interaction_type(depth, active)
            specular = (int_type == InteractionType.SPECULAR) & active
            none = (int_type == InteractionType.NONE) & active
            # Deactivate the ray if no interaction
            active &= ~none

            # Flag indicating if this specular interaction is part of the
            # specular suffix
            in_sf = depth >= sf_start_depth
            specular &= in_sf

            # Read the normal to the reflecting plate and the vertex lying on
            # the reflecting plate
            vertex = paths_base.get_vertex(depth, specular)
            normal = paths.get_primitive_props(depth,
                                               return_normal=True,
                                               return_vertices=False,
                                               active=specular)

            # Compute the image of the current vertex with respect to the
            # reflecting plate
            source_images = point_plane_reflection(source_images,
                                                   normal,
                                                   vertex,
                                                   specular)

            # Use path vertices as a buffer to store the source images
            paths.set_vertex(depth, source_images, in_sf)

            depth += 1
            active &= depth <= max_depth

    @dr.syntax
    def _backtrack(self,
                   mi_scene: mi.Scene,
                   paths: PathsBuffer,
                   paths_targets: mi.Point3f,
                   sf_source: mi.Point3f,
                   sf_start_depth: mi.UInt,
                   valid_candidate: mi.Bool) -> mi.Bool:
        r"""
        Traces backwards from the targets to the images of the sources

        This function assumes that ``_compute_images()`` has already been
        executed.

        When backtracking, this function does the following:
        * It computes the path vertices, i.e., the final intersection point
        of valid candidates with the scene
        * It updates the ``valid_candidate`` array by flagging as invalid
        candidates that are occluded

        :param mi_scene: Mitsuba scene
        :param paths: Candidate paths
        :param paths_targets: Positions of the targets
        :param sf_source: Positions of the sources of the specular suffixes
        :param sf_start_depth: Depths at which the specular chain starts
        :param valid_candidate: Flags specifying candidates marked as valid

        :return: Updated ``valid_candidate`` array
        """

        max_depth = paths.max_depth

        active = dr.copy(valid_candidate)
        depth = dr.full(mi.UInt, max_depth, paths.buffer_size)
        was_none = dr.full(mi.Bool, True, paths.buffer_size)
        vertex = dr.copy(paths_targets)
        normal = dr.zeros(mi.Normal3f, paths.buffer_size)
        # Temporary copy to avoid write-after-read in the symbolic loop.
        paths_base = PathsBufferBase.from_paths_buffer(paths)

        while dr.hint(active, mode=self.loop_mode):

            int_type = paths.get_interaction_type(depth, active)
            # If the depth is `sf_start_depth-1`, then this is the first segment
            # of the specular suffix
            first_segment = depth == (sf_start_depth-1)

            specular_reflection = int_type == InteractionType.SPECULAR
            refraction = int_type == InteractionType.REFRACTION
            none = (int_type == InteractionType.NONE) & ~first_segment
            valid_inter = active & ~none

            # Shoot a ray towards:
            # - The image of the source if the intersection is a specular
            # reflection or refraction
            # - The source of the specular suffix if this is the first segment
            # of the specular suffix
            ray_target = paths_base.get_vertex(depth, valid_inter)
            ray_target = dr.select(first_segment, sf_source, ray_target)

            # Spawn a ray from the current vertex towards `ray_target`.
            # To avoid ray leakage in case where the path vertex is located
            # exactly at the intersection between two perpendicular shapes, a
            # small offset is added to the ray target along the corresponding
            # normal
            normal_image = paths_base.get_primitive_props(
                depth, return_normal=True, return_vertices=False,
                active=valid_inter
            )
            k = vertex - ray_target
            ray = spawn_ray_to(vertex,
                               offset_p(ray_target, k, normal_image),
                               normal)

            # The intersection point of `ray` with the scene is the path next
            # vertex if this is a specular reflection or refraction.
            # For the first segment of the suffix, any intersection means the
            # source is occluded.
            should_not_intersect = first_segment
            si_scene = mi_scene.ray_intersect(ray,
                                              ray_flags=mi.RayFlags.Minimal,
                                              coherent=True,
                                              active=valid_inter)
            next_vertex = dr.select(should_not_intersect, ray_target,
                                    si_scene.p)
            # Check that the intersection is valid.
            valid_inter &= si_scene.is_valid() ^ should_not_intersect

            # Flag indicating that the expected triangle was hit
            expected_shape = paths_base.get_shape(depth, valid_inter)
            si_shape = dr.reinterpret_array(mi.UInt, si_scene.shape)
            expected_prim_ind = paths_base.get_primitive(depth, valid_inter)
            hit_expected_prim = (expected_shape == si_shape) \
                                & (expected_prim_ind == si_scene.prim_index)

            # If a different triangle was hit, check if it would have produced
            # the same image (i.e. it is coplanar with the original)
            check_image = valid_inter & ~hit_expected_prim & specular_reflection

            expected_shape_ptr = dr.reinterpret_array(mi.MeshPtr,
                                                      expected_shape)
            expected_f = expected_shape_ptr.face_indices(expected_prim_ind,
                                                         check_image)
            expected_n = expected_shape_ptr.face_normal(expected_prim_ind,
                                                        check_image)
            # Any point on the triangle works
            expected_p = expected_shape_ptr.vertex_position(expected_f[0],
                                                            check_image)

            # Compute the image of the image through the expected triangle
            exp_img_img = ray_target - 2 *\
                dr.dot((ray_target - expected_p), expected_n) * expected_n
            # Same through the triangle actually being hit
            new_img_img = ray_target - 2 *\
                dr.dot((ray_target - si_scene.p), si_scene.n) * si_scene.n

            valid_specular = specular_reflection & (
                hit_expected_prim
                | (dr.squared_norm(exp_img_img - new_img_img) < 1e-4)
            )
            # Refractions don't affect path geometry, so they are always valid
            valid_inter &= valid_specular | refraction | should_not_intersect

            # Check the segment length is above a pre-defined threshold
            length = dr.select(should_not_intersect,
                               dr.norm(vertex - next_vertex),
                               si_scene.t)
            valid_inter &= (length > MIN_SEGMENT_LENGTH)

            # If the intersection if valid, then stores the interaction point
            # as the path vertex, update the direction of arrival as well as
            # the shape and primitive indices
            paths.set_vertex(depth, next_vertex, valid_inter)
            # Update the angle of arrival if this is the last segment of the
            # specular suffix
            paths.set_angles_rx(ray.d, valid_inter & was_none)
            paths.set_shape(depth, si_shape,
                            valid_inter & ~should_not_intersect)
            paths.set_primitive(depth, si_scene.prim_index,
                                valid_inter & ~should_not_intersect)
            # If the intersection is not valid, discard the candidate
            # If there was no intersection (none == True), then we did not
            # enter yet the specular suffix
            valid_candidate &= valid_inter | none

            active &= (depth >= sf_start_depth) & valid_candidate
            depth -= 1

            vertex = dr.select(valid_inter, next_vertex, vertex)
            normal = dr.select(valid_inter, si_scene.n, normal)
            was_none = dr.copy(none)

        # If the candidate is valid, then update the direction of depature
        specular_chain = (sf_start_depth == 1) & valid_candidate
        vertex = paths.get_vertex(sf_start_depth, specular_chain)
        d_tx = dr.normalize(vertex - sf_source)
        paths.set_angles_tx(d_tx, specular_chain)

        return valid_candidate
