import mitsuba as mi
import drjit as dr
from typing import Callable, Tuple, List

from scipy.constants import speed_of_light

from round2.physical_prior.geometry_prior_engine.utils import (
    rotation_matrix,
    jones_vec_dot,
    spectrum_to_matrix_4f,
)
from round2.physical_prior.geometry_prior_engine.array_model import (
    element_response_to_world,
)
from round2.physical_prior.geometry_prior_engine.constants import InteractionType
from .component_buffer import ComponentBuffer


class ComponentResponseEvaluator:
    def __call__(
        self,
        wavelength: mi.Float | float,
        components: ComponentBuffer,
        samples_per_src: int,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f,
        src_orientations: mi.Point3f,
        tgt_orientations: mi.Point3f,
        src_antenna_patterns: List[
            Callable[[mi.Float, mi.Float], Tuple[mi.Complex2f, mi.Complex2f]]
        ],
        tgt_antenna_patterns: List[
            Callable[[mi.Float, mi.Float], Tuple[mi.Complex2f, mi.Complex2f]]
        ],
    ):

        if components.buffer_size == 0:
            components.a = dr.zeros(mi.Complex2f, 0)
            components.tau = dr.zeros(mi.Float, 0)
            return components

        with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, False):
            self._evaluate_component_responses(
                wavelength,
                components,
                samples_per_src,
                src_positions,
                tgt_positions,
                src_orientations,
                tgt_orientations,
                src_antenna_patterns,
                tgt_antenna_patterns,
            )

        return components

    @dr.syntax
    def _evaluate_component_responses(
        self,
        wavelength: mi.Float | float,
        components: ComponentBuffer,
        samples_per_src: int,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f,
        src_orientations: mi.Point3f,
        tgt_orientations: mi.Point3f,
        src_antenna_patterns: List[
            Callable[[mi.Float, mi.Float], Tuple[mi.Complex2f, mi.Complex2f]]
        ],
        tgt_antenna_patterns: List[
            Callable[[mi.Float, mi.Float], Tuple[mi.Complex2f, mi.Complex2f]]
        ],
    ):
        num_components = components.buffer_size
        max_depth = components.max_depth

        component_source_position = dr.gather(mi.Point3f, src_positions, components.source_indices)
        component_target_position = dr.gather(mi.Point3f, tgt_positions, components.target_indices)

        component_source_orientation = dr.gather(mi.Point3f, src_orientations, components.source_indices)
        src_to_world = rotation_matrix(component_source_orientation)
        component_target_orientation = dr.gather(mi.Point3f, tgt_orientations, components.target_indices)
        tgt_to_world = rotation_matrix(component_target_orientation)

        depth = dr.full(mi.UInt, 1, num_components)

        interaction_type = components.get_interaction_type(1, True)

        active = (
            (depth <= max_depth)
            & (interaction_type != InteractionType.NONE)
            & components.valid
        )

        prev_vertex = component_source_position

        vertex = dr.select(~active, component_target_position, components.get_vertex(1, True))

        next_vertex = dr.zeros(mi.Point3f, num_components)

        ki_world = mi.Vector3f(vertex - prev_vertex)
        component_length = dr.norm(ki_world)
        ki_world *= dr.rcp(component_length)

        ko_world = dr.zeros(mi.Vector3f, num_components)

        e_fields = [
            element_response_to_world(
                src_antenna_pattern, src_to_world, ki_world, incoming=False
            )
            for src_antenna_pattern in src_antenna_patterns
        ]

        solid_angle = dr.full(
            mi.Float, 4.0 * dr.pi * dr.rcp(samples_per_src), num_components
        )

        geometric_spreading_length = dr.copy(component_length)

        while dr.hint(active, mode="symbolic", exclude=[wavelength]):

            last_depth = depth == max_depth

            gather_next = active & ~last_depth

            diffuse = active & (interaction_type == InteractionType.DIFFUSE)

            next_interaction_type = components.get_interaction_type(depth + 1, gather_next)

            next_is_none = next_interaction_type == InteractionType.NONE

            last_interaction = active & (next_is_none | last_depth)

            next_vertex = dr.select(
                last_interaction, component_target_position, components.get_vertex(depth + 1, gather_next)
            )

            ko_world = dr.select(active, mi.Vector3f(next_vertex - vertex), ko_world)
            length = dr.norm(ko_world)
            ko_world *= dr.rcp(length)

            shape = components.get_shape(depth, active)
            shape = dr.reinterpret_array(mi.ShapePtr, shape)

            e_fields, solid_angle = self._update_polarized_response(
                shape,
                components,
                depth,
                interaction_type,
                ki_world,
                ko_world,
                e_fields,
                solid_angle,
                active,
            )

            component_length += dr.select(active, length, 0.0)

            geometric_spreading_length = dr.select(diffuse, 0.0, geometric_spreading_length)
            geometric_spreading_length += dr.select(active, length, 0.0)

            solid_angle = dr.select(diffuse, dr.two_pi, solid_angle)

            depth += 1
            ki_world = dr.select(active, ko_world, ki_world)
            active &= (depth <= max_depth) & ~next_is_none
            prev_vertex = dr.copy(vertex)
            vertex = dr.copy(next_vertex)
            interaction_type = dr.copy(next_interaction_type)

        spreading_factor = dr.rcp(geometric_spreading_length)

        wl_scaling = wavelength * dr.rcp(4.0 * dr.pi)

        tgt_patterns = [
            element_response_to_world(
                tgt_antenna_pattern, tgt_to_world, -ki_world, incoming=True
            )
            for tgt_antenna_pattern in tgt_antenna_patterns
        ]

        a = []
        valid_a = mi.Bool(False)
        for tgt_pattern in tgt_patterns:
            a.append([])
            for e_field in e_fields:
                a_ = jones_vec_dot(tgt_pattern, e_field)
                a_ *= spreading_factor * wl_scaling
                valid_a |= dr.abs(a_) > 0.0
                a[-1].append(a_)

        components.valid &= valid_a

        tau = component_length * dr.rcp(speed_of_light)

        components.a = a
        components.tau = tau

    def _update_polarized_response(
        self,
        shape: mi.ShapePtr,
        components: ComponentBuffer,
        depth: mi.UInt,
        interaction_type: mi.UInt,
        ki_world: mi.Vector3f,
        ko_world: mi.Vector3f,
        e_fields: List[mi.Vector4f],
        solid_angle: mi.Float,
        active: mi.Bool,
    ) -> Tuple[mi.Matrix4f, mi.Float]:


        rm = shape.bsdf()

        num_components = components.buffer_size

        normal_world = components.get_primitive_props(
            depth, return_normal=True, return_vertices=False, active=active
        )

        si = dr.zeros(mi.SurfaceInteraction3f, num_components)
        ctx = mi.BSDFContext(mode=mi.TransportMode.Importance, type_mask=0, component=0)

        normal_world *= dr.sign(dr.dot(normal_world, -ki_world))
        si.n = normal_world
        si.sh_frame.n = normal_world
        si.initialize_sh_frame()

        si.wi = si.to_local(ki_world)

        si.p = components.get_vertex(depth, active)

        si.shape = shape

        si.prim_index = components.get_primitive(depth, active)

        si.dp_du = mi.Vector3f(
            0.0, 0.0, dr.reinterpret_array(mi.Float, interaction_type)
        )

        probs = components.get_prob(depth, active)

        solid_angle[active] *= dr.rcp(probs)

        for i, e_field in enumerate(e_fields):

            si.duv_dx = mi.Vector2f(e_field.x, e_field.y)
            si.duv_dy = mi.Vector2f(e_field.z, e_field.w)

            si.t = solid_angle

            jones_mat = rm.eval(ctx, si, ko_world, active)
            jones_mat = spectrum_to_matrix_4f(jones_mat)

            e_fields[i] = dr.select(active, jones_mat @ e_field, e_field)

        return e_fields, solid_angle
