import drjit as dr

from .multistep_interaction_sampler import MultistepInteractionSampler
from .reflection_geometry_builder import ReflectionGeometryBuilder
from .component_response import ComponentResponseEvaluator
from .propagation_components import PropagationComponents
from round2.physical_prior.geometry_prior_engine.scene import Scene


class ScenePriorBuilder:
    def __init__(self):

        self._candidate_generator = MultistepInteractionSampler()

        self._specular_constructor = ReflectionGeometryBuilder()

        self._component_response = ComponentResponseEvaluator()

    def __call__(
        self,
        scene: Scene,
        max_depth: int = 3,
        max_components_per_source: int = 1000000,
        samples_per_src: int = 1000000,
        diffuse_reflection: bool = False,
        seed: int = 42,
    ) -> PropagationComponents:
        scene.validate()

        src_positions, src_orientations, rel_ant_positions_tx = scene.sources()
        tgt_positions, tgt_orientations, rel_ant_positions_rx = scene.targets()

        src_antenna_patterns = scene.tx_array.elements
        tgt_antenna_patterns = scene.rx_array.elements
        dr.make_opaque(src_positions, tgt_positions, src_orientations, tgt_orientations)

        component_buffer = self._candidate_generator(
            mi_scene=scene.mi_scene,
            src_positions=src_positions,
            tgt_positions=tgt_positions,
            samples_per_src=samples_per_src,
            max_components_per_source=max_components_per_source,
            max_depth=max_depth,
            diffuse_reflection=diffuse_reflection,
            seed=seed,
        )

        component_buffer.schedule()
        dr.eval()

        component_buffer.shrink()

        component_buffer.detach_geometry()

        component_buffer = self._specular_constructor(
            scene=scene.mi_scene,
            components=component_buffer,
            src_positions=src_positions,
            tgt_positions=tgt_positions,
        )

        component_buffer = self._component_response(
            wavelength=scene.wavelength,
            components=component_buffer,
            samples_per_src=samples_per_src,
            src_positions=src_positions,
            tgt_positions=tgt_positions,
            src_orientations=src_orientations,
            tgt_orientations=tgt_orientations,
            src_antenna_patterns=src_antenna_patterns,
            tgt_antenna_patterns=tgt_antenna_patterns,
        )

        component_buffer.discard_invalid()

        components = PropagationComponents(
            scene,
            component_buffer,
            rel_ant_positions_tx,
            rel_ant_positions_rx,
        )

        return components
