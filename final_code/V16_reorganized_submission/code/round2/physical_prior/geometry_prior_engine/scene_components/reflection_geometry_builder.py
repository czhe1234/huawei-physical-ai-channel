import drjit as dr
import mitsuba as mi
from typing import Tuple

from round2.physical_prior.geometry_prior_engine.utils import (
    offset_p,
    mirror_point_across_plane,
    build_segment_scene_query,
)
from round2.physical_prior.geometry_prior_engine.constants import (
    InteractionType,
    MIN_SEGMENT_LENGTH,
)
from .component_buffer import ComponentBuffer, ComponentBufferBase


class ReflectionGeometryBuilder:
    def __call__(
        self,
        scene: mi.Scene,
        components: ComponentBuffer,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f,
    ) -> ComponentBuffer:

        if components.buffer_size == 0:
            return components

        valid_candidate = ~components.valid

        component_sources, component_targets = self._gather_component_endpoints(
            components, src_positions, tgt_positions, valid_candidate
        )

        sequence_start_depth = self._coherent_sequence_start_depth(components, valid_candidate)

        sequence_origin = self._select_sequence_origin(
            components, component_sources, sequence_start_depth, valid_candidate
        )

        self._construct_mirrored_geometry(components, sequence_origin, sequence_start_depth, valid_candidate)

        valid_candidate = self._validate_interaction_geometry(
            scene, components, component_targets, sequence_origin, sequence_start_depth, valid_candidate
        )

        components.valid |= valid_candidate
        dr.schedule(components.valid)

        return components

    def _gather_component_endpoints(
        self,
        components,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f,
        valid_candidate: mi.Bool,
    ) -> Tuple[mi.Point3f, mi.Point3f]:

        component_sources = dr.gather(
            mi.Point3f, src_positions, components.source_indices, valid_candidate
        )
        component_targets = dr.gather(
            mi.Point3f, tgt_positions, components.target_indices, valid_candidate
        )
        return component_sources, component_targets

    @dr.syntax
    def _coherent_sequence_start_depth(
        self, components: ComponentBuffer, valid_candidate: mi.Bool
    ) -> mi.UInt:

        max_depth = components.max_depth
        max_components = components.buffer_size

        sequence_start_depth = dr.full(mi.UInt, max_depth, max_components)

        active = dr.copy(valid_candidate)
        depth = mi.UInt(max_depth)
        while dr.hint(active, mode="symbolic"):

            int_type = components.get_interaction_type(depth, active)
            specular = int_type == InteractionType.SPECULAR
            refraction = int_type == InteractionType.REFRACTION
            none = int_type == InteractionType.NONE

            active &= specular | refraction | none

            sequence_start_depth = dr.select(active, depth, sequence_start_depth)

            depth -= 1
            active &= depth > 0

        return sequence_start_depth

    def _select_sequence_origin(
        self,
        components: ComponentBuffer,
        component_sources: mi.Point3f,
        sequence_start_depth: mi.UInt,
        valid_candidate: mi.Bool,
    ) -> mi.Point3f:

        sequence_origin = sequence_start_depth == 1

        previous_interaction_depth = sequence_start_depth - 1
        previous_interaction_depth = dr.select(sequence_origin, 1, previous_interaction_depth)

        vertex = components.get_vertex(previous_interaction_depth, valid_candidate)
        sequence_origin = dr.select(sequence_origin, component_sources, vertex)

        return sequence_origin

    @dr.syntax
    def _construct_mirrored_geometry(
        self,
        components: ComponentBuffer,
        sequence_origin: mi.Point3f,
        sequence_start_depth: mi.UInt,
        valid_candidate: mi.Bool,
    ) -> None:

        max_depth = components.max_depth
        num_components = components.buffer_size

        mirrored_origins = dr.copy(sequence_origin)

        normal = dr.zeros(mi.Vector3f, num_components)
        active = dr.copy(valid_candidate)
        depth = dr.full(mi.UInt, 1, num_components)

        component_geometry = ComponentBufferBase.from_component_buffer(components)

        while dr.hint(active, mode="symbolic"):

            int_type = components.get_interaction_type(depth, active)
            specular = (int_type == InteractionType.SPECULAR) & active
            none = (int_type == InteractionType.NONE) & active

            active &= ~none

            in_sequence = depth >= sequence_start_depth
            specular &= in_sequence

            vertex = component_geometry.get_vertex(depth, specular)
            normal = components.get_primitive_props(
                depth, return_normal=True, return_vertices=False, active=specular
            )

            mirrored_origins = mirror_point_across_plane(
                mirrored_origins, normal, vertex, specular
            )

            components.set_vertex(depth, mirrored_origins, in_sequence)

            depth += 1
            active &= depth <= max_depth

    @dr.syntax
    def _validate_interaction_geometry(
        self,
        mi_scene: mi.Scene,
        components: ComponentBuffer,
        component_targets: mi.Point3f,
        sequence_origin: mi.Point3f,
        sequence_start_depth: mi.UInt,
        valid_candidate: mi.Bool,
    ) -> mi.Bool:

        max_depth = components.max_depth

        active = dr.copy(valid_candidate)
        depth = dr.full(mi.UInt, max_depth, components.buffer_size)
        was_none = dr.full(mi.Bool, True, components.buffer_size)
        vertex = dr.copy(component_targets)
        normal = dr.zeros(mi.Normal3f, components.buffer_size)

        component_geometry = ComponentBufferBase.from_component_buffer(components)

        while dr.hint(active, mode="symbolic"):

            int_type = components.get_interaction_type(depth, active)

            first_segment = depth == (sequence_start_depth - 1)

            specular_reflection = int_type == InteractionType.SPECULAR
            refraction = int_type == InteractionType.REFRACTION
            none = (int_type == InteractionType.NONE) & ~first_segment
            valid_inter = active & ~none

            segment_target = component_geometry.get_vertex(depth, valid_inter)
            segment_target = dr.select(first_segment, sequence_origin, segment_target)

            interface_normal = component_geometry.get_primitive_props(
                depth, return_normal=True, return_vertices=False, active=valid_inter
            )
            k = vertex - segment_target
            query = build_segment_scene_query(vertex, offset_p(segment_target, k, interface_normal), normal)

            should_not_intersect = first_segment
            si_scene = mi_scene.ray_intersect(
                query, ray_flags=mi.RayFlags.Minimal, coherent=True, active=valid_inter
            )
            next_vertex = dr.select(should_not_intersect, segment_target, si_scene.p)

            valid_inter &= si_scene.is_valid() ^ should_not_intersect

            expected_shape = component_geometry.get_shape(depth, valid_inter)
            si_shape = dr.reinterpret_array(mi.UInt, si_scene.shape)
            expected_prim_ind = component_geometry.get_primitive(depth, valid_inter)
            hit_expected_prim = (expected_shape == si_shape) & (
                expected_prim_ind == si_scene.prim_index
            )

            compare_mirrored_geometry = valid_inter & ~hit_expected_prim & specular_reflection

            expected_shape_ptr = dr.reinterpret_array(mi.MeshPtr, expected_shape)
            expected_f = expected_shape_ptr.face_indices(expected_prim_ind, compare_mirrored_geometry)
            expected_n = expected_shape_ptr.face_normal(expected_prim_ind, compare_mirrored_geometry)

            expected_p = expected_shape_ptr.vertex_position(expected_f[0], compare_mirrored_geometry)

            expected_mirror_point = (
                segment_target
                - 2 * dr.dot((segment_target - expected_p), expected_n) * expected_n
            )

            observed_mirror_point = (
                segment_target
                - 2 * dr.dot((segment_target - si_scene.p), si_scene.n) * si_scene.n
            )

            valid_coherent_reflection = specular_reflection & (
                hit_expected_prim | (dr.squared_norm(expected_mirror_point - observed_mirror_point) < 1e-4)
            )

            valid_inter &= valid_coherent_reflection | refraction | should_not_intersect

            length = dr.select(
                should_not_intersect, dr.norm(vertex - next_vertex), si_scene.t
            )
            valid_inter &= length > MIN_SEGMENT_LENGTH

            components.set_vertex(depth, next_vertex, valid_inter)

            components.set_angles_rx(query.d, valid_inter & was_none)
            components.set_shape(depth, si_shape, valid_inter & ~should_not_intersect)
            components.set_primitive(
                depth, si_scene.prim_index, valid_inter & ~should_not_intersect
            )

            valid_candidate &= valid_inter | none

            active &= (depth >= sequence_start_depth) & valid_candidate
            depth -= 1

            vertex = dr.select(valid_inter, next_vertex, vertex)
            normal = dr.select(valid_inter, si_scene.n, normal)
            was_none = dr.copy(none)

        coherent_sequence = (sequence_start_depth == 1) & valid_candidate
        vertex = components.get_vertex(sequence_start_depth, coherent_sequence)
        d_tx = dr.normalize(vertex - sequence_origin)
        components.set_angles_tx(d_tx, coherent_sequence)

        return valid_candidate
