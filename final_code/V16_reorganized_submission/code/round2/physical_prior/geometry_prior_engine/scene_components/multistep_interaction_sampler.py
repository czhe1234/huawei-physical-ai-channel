import mitsuba as mi
import drjit as dr
from typing import Tuple

from round2.physical_prior.geometry_prior_engine.constants import (
    InteractionType,
    MIN_SEGMENT_LENGTH,
    NO_JONES_MATRIX,
)
from round2.physical_prior.geometry_prior_engine.utils import (
    PlaneHasher,
    fibonacci_lattice,
    hash_fnv1a,
    build_directional_scene_queries,
    build_segment_scene_query,
)
from .interaction_samples import InteractionSamples
from .component_buffer import ComponentBuffer


class MultistepInteractionSampler:
    MIN_SPEC_COUNT_SIZE = int(1e6)

    def __init__(self):

        self._sampler = mi.load_dict({"type": "independent"})
        self.plane_hash_functions = [PlaneHasher(op="round"), PlaneHasher(op="floor")]

    def __call__(
        self,
        mi_scene: mi.Scene,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f,
        samples_per_src: int,
        max_components_per_source: int,
        max_depth: int,
        diffuse_reflection: bool,
        seed: int = 1,
    ) -> ComponentBuffer:
        num_sources = dr.shape(src_positions)[1]
        num_samples = samples_per_src * num_sources
        max_components = max_components_per_source * num_sources

        self._sampler.seed(seed, num_samples)

        components = ComponentBuffer(max_components, max_depth)

        component_counter_per_source = dr.zeros(mi.UInt, num_sources)

        self._evaluate_direct_visibility(
            mi_scene, src_positions, tgt_positions, components, component_counter_per_source
        )

        if max_depth > 0:
            self._sample_multistep_interactions(
                mi_scene,
                src_positions,
                tgt_positions,
                components,
                samples_per_src,
                max_components_per_source,
                max_depth,
                component_counter_per_source,
                diffuse_reflection,
            )

        return components

    @dr.syntax
    def _evaluate_direct_visibility(
        self,
        mi_scene: mi.Scene,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f,
        components: ComponentBuffer,
        component_counter_per_source: mi.UInt,
    ):


        num_src = dr.width(src_positions)
        num_tgt = dr.width(tgt_positions)

        samples_data = InteractionSamples(num_src, num_tgt, 0)

        tgt_indices = dr.arange(mi.UInt, num_tgt)
        tgt_indices = dr.tile(tgt_indices, num_src)

        origins = dr.repeat(src_positions, num_tgt)
        targets = dr.tile(tgt_positions, num_src)

        length = dr.norm(targets - origins)
        valid = length > MIN_SEGMENT_LENGTH

        queries = build_segment_scene_query(origins, targets)
        valid &= ~mi_scene.ray_test(queries, active=valid)

        component_index = dr.scatter_inc(components.component_counter, mi.UInt(0), valid)
        components.add_components(
            mi.UInt(0),
            component_index,
            samples_data,
            valid,
            tgt_indices,
            queries.d,
            -queries.d,
            valid,
        )

        dr.scatter_inc(component_counter_per_source, samples_data.src_indices, valid)

    def _sample_multistep_interactions(
        self,
        mi_scene: mi.Scene,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f,
        components: ComponentBuffer,
        samples_per_src: int,
        max_components_per_source: int,
        max_depth: int,
        component_counter_per_source: mi.UInt,
        diffuse_reflection: bool,
    ):


        with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, False):
            self._sample_interaction_sequence(
                mi_scene,
                src_positions,
                tgt_positions,
                samples_per_src,
                max_components_per_source,
                max_depth,
                components,
                component_counter_per_source,
                diffuse_reflection,
                self._sampler,
            )

    @dr.syntax
    def _sample_interaction_sequence(
        self,
        mi_scene: mi.Scene,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f,
        samples_per_src: int,
        max_components_per_source: int,
        max_depth: int,
        components: ComponentBuffer,
        component_counter_per_source: mi.UInt,
        diffuse_reflection_enabled: bool,
        sampler: mi.Sampler,
    ):


        num_sources = dr.shape(src_positions)[1]
        num_targets = dr.shape(tgt_positions)[1]
        num_samples = samples_per_src * num_sources

        samples_data = InteractionSamples(num_sources, samples_per_src, max_depth)

        query = build_directional_scene_queries(fibonacci_lattice, samples_per_src, src_positions)

        k_tx = dr.copy(query.d)

        coherent_sequence = dr.full(mi.Bool, True, num_samples)

        num_hashes = len(self.plane_hash_functions)
        hashes = [dr.zeros(mi.UInt64, num_samples) for _ in range(num_hashes)]

        spec_counter_size = dr.maximum(
            max_components_per_source, MultistepInteractionSampler.MIN_SPEC_COUNT_SIZE
        )
        coherent_sequence_counters = [
            dr.zeros(mi.UInt, spec_counter_size * num_sources)
            for _ in range(num_hashes)
        ]

        depth = dr.full(mi.UInt, 1, num_samples)

        active = dr.full(mi.Mask, True, num_samples)

        loc_en_inter = dr.full(mi.UInt, 0, num_samples)
        loc_en_inter |= InteractionType.SPECULAR
        if diffuse_reflection_enabled:
            loc_en_inter |= InteractionType.DIFFUSE
        loc_en_inter |= InteractionType.REFRACTION

        while dr.hint(
            active,
            label="sample_multistep_interactions",
            exclude=[
                coherent_sequence_counters,
                component_counter_per_source,
            ],
        ):

            si_scene = mi_scene.ray_intersect(
                query, coherent=True, ray_flags=mi.RayFlags.Minimal, active=active
            )

            active &= si_scene.is_valid()

            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()

            sampler.next_2d()
            s, n = self._sample_scene_response(
                si_scene,
                query.d,
                sample1,
                sample2,
                loc_en_inter,
                active,
            )

            k_world = s.wo

            int_type = dr.select(active, s.sampled_component, InteractionType.NONE)

            active &= int_type != InteractionType.NONE

            specular = int_type == InteractionType.SPECULAR

            transmission = int_type == InteractionType.REFRACTION

            diffuse = int_type == InteractionType.DIFFUSE

            coherent_sequence &= active & (specular | transmission)

            samples_data.insert(
                depth,
                int_type,
                si_scene.shape,
                si_scene.prim_index,
                si_scene.p,
                s.pdf,
                active,
            )

            for i in range(num_hashes):
                plane_hash = self.plane_hash_functions[i](si_scene.n, si_scene.p)
                inter_hash = dr.select(
                    int_type == InteractionType.SPECULAR, plane_hash, int_type
                )
                hashes[i] = hash_fnv1a(inter_hash, h=hashes[i])

            t = mi.UInt(0)
            while dr.hint(
                t < num_targets,
                label="sample_interaction_targets",
                exclude=[coherent_sequence_counters, component_counter_per_source],
            ):

                tgt_position = dr.gather(mi.Point3f, tgt_positions, t)

                target_query = si_scene.spawn_ray_to(tgt_position)
                target_blocked = mi_scene.ray_test(target_query, active=active)
                target_visible = ~target_blocked

                target_incident_side = dr.dot(n, target_query.d) > 0.0
                valid = target_visible & diffuse & target_incident_side

                new_coherent_candidate = coherent_sequence & target_visible

                for i in range(num_hashes):
                    component_target_hash = hash_fnv1a(t, h=hashes[i])
                    counter_ind = component_target_hash % spec_counter_size
                    counter_ind += spec_counter_size * samples_data.src_indices

                    samples_counter = dr.scatter_inc(
                        coherent_sequence_counters[i], counter_ind, new_coherent_candidate
                    )
                    new_coherent_candidate &= samples_counter == 0

                store = active & (valid | new_coherent_candidate)

                component_count_for_source = dr.scatter_inc(
                    component_counter_per_source, samples_data.src_indices, store
                )

                store &= component_count_for_source < max_components_per_source

                component_index = dr.scatter_inc(components.component_counter, mi.UInt(0), store)

                components.add_components(
                    depth, component_index, samples_data, valid, t, k_tx, -target_query.d, store
                )

                t += 1

            depth += 1
            active &= depth <= max_depth

            query = si_scene.spawn_ray(d=k_world)

            coherent_sequence |= diffuse

    def _sample_scene_response(
        self,
        si: mi.SurfaceInteraction3f,
        k_world: mi.Vector3f,
        sample1: mi.Float,
        sample2: mi.Point2f,
        loc_en_inter: mi.UInt,
        active: mi.Bool,
    ) -> Tuple[mi.BSDFSample3f, mi.Normal3f]:


        normal_world = si.n * dr.sign(dr.dot(si.n, -k_world))
        si.sh_frame.n = normal_world
        si.initialize_sh_frame()
        si.n = normal_world

        ctx = mi.BSDFContext(mode=mi.TransportMode.Importance, type_mask=0, component=0)

        ctx.component |= NO_JONES_MATRIX

        si.wi = si.to_local(k_world)

        si.dp_du = mi.Vector3f(1.0, 1.0, dr.reinterpret_array(mi.Float, loc_en_inter))

        sample, _ = si.bsdf().sample(ctx, si, sample1, sample2, active)

        return sample, normal_world
