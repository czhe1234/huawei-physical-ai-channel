import mitsuba as mi
import drjit as dr
from dataclasses import dataclass
from typing import Tuple


class InteractionSamples:

    @dataclass
    class InteractionSampleFields:

        interaction_type: mi.UInt
        shape: mi.UInt
        primitive: mi.UInt
        vertex: mi.Point3f
        prob: mi.Float

    def __init__(self, num_sources: int, samples_per_src: int, max_depth: int):

        array_size = max(1, max_depth)

        src_indices = dr.arange(mi.UInt, 0, num_sources)
        src_indices = dr.repeat(src_indices, samples_per_src)
        self._src_indices = src_indices

        self._local_mem = dr.alloc_local(InteractionSamples.InteractionSampleFields, array_size)

    def insert(
        self,
        depth: mi.UInt,
        interaction_types: mi.UInt,
        shapes: mi.ShapePtr,
        primitives: mi.UInt,
        vertices: mi.Point3f,
        probs: mi.Float,
        active: mi.Mask,
    ):


        index = depth - 1

        shapes = dr.reinterpret_array(mi.UInt, shapes)

        data = InteractionSamples.InteractionSampleFields(
            interaction_types, shapes, primitives, vertices, probs
        )
        self._local_mem.write(data, index, active=active)

    def get(
        self, depth: mi.UInt, active: mi.Mask
    ) -> Tuple[mi.UInt, mi.UInt, mi.UInt, mi.Point3f, mi.Float]:

        index = depth - 1
        data = self._local_mem.read(index, active=active)

        interaction_types = data.interaction_type
        shapes = data.shape
        primitives = data.primitive
        vertices = data.vertex
        probs = data.prob

        return interaction_types, shapes, primitives, vertices, probs

    @property
    def src_indices(self):
        return self._src_indices
