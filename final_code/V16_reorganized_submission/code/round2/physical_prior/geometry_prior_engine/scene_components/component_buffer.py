import mitsuba as mi
import drjit as dr
from typing import Union, Type, Tuple

from round2.physical_prior.geometry_prior_engine.constants import (
    InteractionType,
    INVALID_SHAPE,
    INVALID_PRIMITIVE,
)
from round2.physical_prior.geometry_prior_engine.utils import theta_phi_from_unit_vec


class ComponentBufferBase:

    def __init__(self, buffer_size: int, max_depth: int):

        depth_dim_size = max(1, max_depth)

        self._max_depth = max_depth
        self._buffer_size = buffer_size
        self._depth_dim_size = depth_dim_size

        self._vertices_x = dr.zeros(mi.TensorXf, [buffer_size, depth_dim_size])
        self._vertices_y = dr.zeros(mi.TensorXf, [buffer_size, depth_dim_size])
        self._vertices_z = dr.zeros(mi.TensorXf, [buffer_size, depth_dim_size])

        self._shapes = dr.full(
            mi.TensorXu, INVALID_SHAPE, [buffer_size, depth_dim_size]
        )

        self._primitives = dr.full(
            mi.TensorXu, INVALID_PRIMITIVE, [buffer_size, depth_dim_size]
        )

        self._tensor_width_lit = dr.width(self._shapes.array)
        self._tensor_width = dr.opaque(mi.UInt32, self._tensor_width_lit)

    @staticmethod
    def from_component_buffer(components: "ComponentBuffer"):

        result = ComponentBufferBase(components.buffer_size, components.max_depth)

        result._vertices_x = dr.copy(components._vertices_x)
        result._vertices_y = dr.copy(components._vertices_y)
        result._vertices_z = dr.copy(components._vertices_z)
        result._shapes = dr.copy(components._shapes)
        result._primitives = dr.copy(components._primitives)
        return result

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def depth_dim_size(self):
        return self._depth_dim_size

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def vertices_x(self):
        return self._vertices_x

    @property
    def vertices_y(self):
        return self._vertices_y

    @property
    def vertices_z(self):
        return self._vertices_z

    @property
    def shapes(self):
        return self._shapes

    @property
    def primitives(self):
        return self._primitives

    def get_vertex(self, depth: mi.UInt, active: mi.Bool) -> mi.Point3f:

        vx = self._gather_depth(mi.Float, self._vertices_x, depth, active)
        vy = self._gather_depth(mi.Float, self._vertices_y, depth, active)
        vz = self._gather_depth(mi.Float, self._vertices_z, depth, active)

        return mi.Point3f(vx, vy, vz)

    def get_shape(self, depth: mi.UInt, active: mi.Bool) -> mi.UInt:

        return self._gather_depth(mi.UInt, self._shapes, depth, active)

    def get_primitive(self, depth: mi.UInt, active: mi.Bool) -> mi.UInt:

        return self._gather_depth(mi.UInt, self._primitives, depth, active)

    def get_primitive_props(
        self,
        depth: mi.UInt,
        return_normal: bool,
        return_vertices: bool,
        active: mi.Bool,
    ) -> (
        mi.Vector3f
        | Tuple[mi.Point3f, mi.Point3f, mi.Point3f]
        | Tuple[mi.Vector3f, mi.Point3f, mi.Point3f, mi.Point3f]
    ):

        valid = active & (depth > 0)

        shape_int = self.get_shape(depth, valid)
        mesh_ptr = dr.reinterpret_array(mi.MeshPtr, shape_int)

        prim_ind = self.get_primitive(depth, valid)

        output = []

        if return_normal:
            normal = mesh_ptr.face_normal(prim_ind, valid)
            if not return_vertices:
                return normal
            output.append(normal)

        if return_vertices:
            v_ind = mesh_ptr.face_indices(prim_ind, valid)
            output.append(mesh_ptr.vertex_position(v_ind.x, valid))
            output.append(mesh_ptr.vertex_position(v_ind.y, valid))
            output.append(mesh_ptr.vertex_position(v_ind.z, valid))
        return tuple(output)

    def _gather_depth(
        self,
        dtype: Type[Union[mi.Float, mi.UInt, mi.Bool]],
        tensor: mi.TensorXf | mi.TensorXu | mi.TensorXb,
        depth: mi.UInt,
        active: mi.Bool,
    ) -> mi.Float | mi.UInt | mi.Bool:

        valid = active & (depth > 0)

        ind = dr.arange(mi.UInt, self.buffer_size) * self.depth_dim_size + depth - 1

        assert dr.width(tensor.array) == self._tensor_width_lit
        ind &= ind < self._tensor_width

        return dr.gather(dtype, tensor.array, ind, valid)


class ComponentBuffer(ComponentBufferBase):

    def __init__(self, buffer_size: int, max_depth: int):
        super().__init__(buffer_size, max_depth)

        self._component_counter = mi.UInt(0)

        self._valid = dr.full(dr.mask_t(mi.Float), False, buffer_size)

        self._src_indices = dr.zeros(mi.UInt, buffer_size)

        self._tgt_indices = dr.zeros(mi.UInt, buffer_size)

        self._theta_t = dr.zeros(mi.Float, buffer_size)
        self._phi_t = dr.zeros(mi.Float, buffer_size)

        self._theta_r = dr.zeros(mi.Float, buffer_size)
        self._phi_r = dr.zeros(mi.Float, buffer_size)

        self._interaction_types = dr.full(
            mi.TensorXu, InteractionType.NONE, [buffer_size, self.depth_dim_size]
        )

        self._probs = dr.zeros(mi.TensorXf, [buffer_size, self.depth_dim_size])

        self._a = None
        self._tau = None

    @property
    def component_counter(self):
        return self._component_counter

    @property
    def valid(self):
        return self._valid

    @valid.setter
    def valid(self, v):
        self._valid = v

    @property
    def source_indices(self):
        return self._src_indices

    @property
    def target_indices(self):
        return self._tgt_indices

    @property
    def theta_t(self):
        return self._theta_t

    @property
    def phi_t(self):
        return self._phi_t

    @property
    def theta_r(self):
        return self._theta_r

    @property
    def phi_r(self):
        return self._phi_r

    @property
    def interaction_types(self):
        return self._interaction_types

    @property
    def probs(self):
        return self._probs

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, v):
        self._a = v

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, v):
        self._tau = v

    def schedule(self) -> None:
        arrays = [
            self._max_depth,
            self._buffer_size,
            self._depth_dim_size,
            self._component_counter,
            self._valid,
            self._src_indices,
            self._tgt_indices,
            self._theta_t,
            self._phi_t,
            self._theta_r,
            self._phi_r,
            self._interaction_types,
            self._vertices_x,
            self._vertices_y,
            self._vertices_z,
            self._shapes,
            self._primitives,
            self._probs,
            self._a,
            self._tau,
        ]
        dr.schedule(*arrays)

    @dr.syntax
    def add_components(
        self,
        depth: mi.UInt,
        indices: mi.UInt,
        interaction_samples: int,
        valid: mi.Bool,
        tgt_index: mi.UInt,
        k_tx: mi.Vector3f,
        k_rx: mi.Vector3f,
        active: mi.Bool,
    ) -> None:


        depth_dim_size = self._depth_dim_size

        dr.scatter(self._valid, valid, indices, active=active)

        dr.scatter(self._src_indices, interaction_samples.src_indices, indices, active)

        dr.scatter(self._tgt_indices, tgt_index, indices, active)

        theta_t, phi_t = theta_phi_from_unit_vec(k_tx)
        dr.scatter(self._theta_t, theta_t, indices, active)
        dr.scatter(self._phi_t, phi_t, indices, active)

        theta_r, phi_r = theta_phi_from_unit_vec(k_rx)
        dr.scatter(self._theta_r, theta_r, indices, active)
        dr.scatter(self._phi_r, phi_r, indices, active)

        d = dr.ones(mi.UInt, dr.width(active))
        while d <= depth:
            interaction_types, shapes, primitives, vertices, probs = interaction_samples.get(
                d, active=active
            )

            indices_t = indices * depth_dim_size + d - 1

            dr.scatter(
                self._interaction_types.array, interaction_types, indices_t, active
            )

            dr.scatter(self._shapes.array, shapes, indices_t, active)

            dr.scatter(self._primitives.array, primitives, indices_t, active)

            dr.scatter(self._vertices_x.array, vertices.x, indices_t, active)
            dr.scatter(self._vertices_y.array, vertices.y, indices_t, active)
            dr.scatter(self._vertices_z.array, vertices.z, indices_t, active)

            dr.scatter(self._probs.array, probs, indices_t, active)

            d += 1

    def shrink(self) -> None:

        num_components = dr.minimum(dr.max(self._component_counter), self._buffer_size)[0]

        depth_dim_size = self._depth_dim_size

        self._valid = dr.reshape(mi.Bool, self._valid, num_components, shrink=True)
        self._src_indices = dr.reshape(
            mi.UInt, self._src_indices, num_components, shrink=True
        )
        self._tgt_indices = dr.reshape(
            mi.UInt, self._tgt_indices, num_components, shrink=True
        )
        if self._a is not None:
            a = []
            for a_ in self._a:
                a1 = []
                for a__ in a_:
                    a1.append(dr.reshape(mi.Complex2f, a__, num_components, shrink=True))
                a.append(a1)
            self._a = a
        if self._tau is not None:
            self._tau = dr.reshape(mi.Float, self._tau, num_components, shrink=True)
        self._theta_t = dr.reshape(mi.Float, self._theta_t, num_components, shrink=True)
        self._phi_t = dr.reshape(mi.Float, self._phi_t, num_components, shrink=True)
        self._theta_r = dr.reshape(mi.Float, self._theta_r, num_components, shrink=True)
        self._phi_r = dr.reshape(mi.Float, self._phi_r, num_components, shrink=True)
        self._interaction_types = mi.TensorXu(
            dr.reshape(
                mi.UInt,
                self._interaction_types.array,
                num_components * depth_dim_size,
                shrink=True,
            ),
            shape=(num_components, depth_dim_size),
        )
        self._vertices_x = mi.TensorXf(
            dr.reshape(
                mi.Float,
                self._vertices_x.array,
                num_components * depth_dim_size,
                shrink=True,
            ),
            shape=(num_components, depth_dim_size),
        )
        self._vertices_y = mi.TensorXf(
            dr.reshape(
                mi.Float,
                self._vertices_y.array,
                num_components * depth_dim_size,
                shrink=True,
            ),
            shape=(num_components, depth_dim_size),
        )
        self._vertices_z = mi.TensorXf(
            dr.reshape(
                mi.Float,
                self._vertices_z.array,
                num_components * depth_dim_size,
                shrink=True,
            ),
            shape=(num_components, depth_dim_size),
        )
        self._shapes = mi.TensorXu(
            dr.reshape(
                mi.UInt, self._shapes.array, num_components * depth_dim_size, shrink=True
            ),
            shape=(num_components, depth_dim_size),
        )
        self._primitives = mi.TensorXu(
            dr.reshape(
                mi.UInt, self._primitives.array, num_components * depth_dim_size, shrink=True
            ),
            shape=(num_components, depth_dim_size),
        )
        self._probs = mi.TensorXf(
            dr.reshape(
                mi.Float, self._probs.array, num_components * depth_dim_size, shrink=True
            ),
            shape=(num_components, depth_dim_size),
        )

        self._buffer_size = num_components
        self._component_counter = num_components
        self._tensor_width_lit = dr.width(self._shapes.array)
        self._tensor_width = dr.opaque(mi.UInt32, self._tensor_width_lit)

    def discard_invalid(self) -> None:

        depth_dim_size = self._depth_dim_size
        self.schedule()

        valid_ind = dr.compress(self._valid)

        num_valid_components = dr.shape(valid_ind)[0]

        depth_ind = dr.tile(dr.arange(mi.UInt, 0, depth_dim_size), num_valid_components)

        if num_valid_components == 0:
            tensor_gind = []
        else:
            tensor_gind = (
                dr.repeat(valid_ind, depth_dim_size) * depth_dim_size + depth_ind
            )

        self._valid = dr.gather(mi.Bool, self._valid, valid_ind)
        self._src_indices = dr.gather(mi.UInt, self._src_indices, valid_ind)
        self._tgt_indices = dr.gather(mi.UInt, self._tgt_indices, valid_ind)
        if self._a is not None:
            a = []
            for a_ in self._a:
                a1 = []
                for a__ in a_:
                    a1.append(dr.gather(mi.Complex2f, a__, valid_ind))
                a.append(a1)
            self._a = a
        if self._tau is not None:
            self._tau = dr.gather(mi.Float, self._tau, valid_ind)
        self._theta_t = dr.gather(mi.Float, self._theta_t, valid_ind)
        self._phi_t = dr.gather(mi.Float, self._phi_t, valid_ind)
        self._theta_r = dr.gather(mi.Float, self._theta_r, valid_ind)
        self._phi_r = dr.gather(mi.Float, self._phi_r, valid_ind)
        self._interaction_types = mi.TensorXu(
            dr.gather(mi.UInt, self._interaction_types.array, tensor_gind),
            shape=(num_valid_components, depth_dim_size),
        )
        self._vertices_x = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_x.array, tensor_gind),
            shape=(num_valid_components, depth_dim_size),
        )
        self._vertices_y = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_y.array, tensor_gind),
            shape=(num_valid_components, depth_dim_size),
        )
        self._vertices_z = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_z.array, tensor_gind),
            shape=(num_valid_components, depth_dim_size),
        )
        self._shapes = mi.TensorXu(
            dr.gather(mi.UInt, self._shapes.array, tensor_gind),
            shape=(num_valid_components, depth_dim_size),
        )
        self._primitives = mi.TensorXu(
            dr.gather(mi.UInt, self._primitives.array, tensor_gind),
            shape=(num_valid_components, depth_dim_size),
        )
        self._probs = mi.TensorXf(
            dr.gather(mi.Float, self._probs.array, tensor_gind),
            shape=(num_valid_components, depth_dim_size),
        )

        self._buffer_size = num_valid_components
        self._component_counter = num_valid_components
        self._tensor_width_lit = dr.width(self._shapes.array)
        self._tensor_width = dr.opaque(mi.UInt32, self._tensor_width_lit)

    def get_interaction_type(self, depth: mi.UInt, active: mi.Bool) -> mi.UInt:

        return self._gather_depth(mi.UInt, self._interaction_types, depth, active)

    def get_prob(self, depth: mi.UInt, active: mi.Bool) -> mi.Float:
        return self._gather_depth(mi.Float, self._probs, depth, active)

    def set_angles_tx(self, k_tx: mi.Vector3f, active: mi.Bool) -> None:

        theta_t, phi_t = theta_phi_from_unit_vec(k_tx)
        indices = dr.arange(mi.UInt, self.buffer_size)
        dr.scatter(self._theta_t, theta_t, indices, active)
        dr.scatter(self._phi_t, phi_t, indices, active)

    def set_angles_rx(self, k_rx: mi.Vector3f, active: mi.Bool) -> None:

        theta_r, phi_r = theta_phi_from_unit_vec(k_rx)
        indices = dr.arange(mi.UInt, self.buffer_size)
        dr.scatter(self._theta_r, theta_r, indices, active)
        dr.scatter(self._phi_r, phi_r, indices, active)

    def set_vertex(self, depth: mi.UInt, value: mi.Point3f, active: mi.Bool) -> None:

        self._write_depth_value(self._vertices_x, depth, value.x, active)
        self._write_depth_value(self._vertices_y, depth, value.y, active)
        self._write_depth_value(self._vertices_z, depth, value.z, active)

    def set_shape(self, depth: mi.UInt, value: mi.UInt, active: mi.Bool) -> None:

        self._write_depth_value(self._shapes, depth, value, active)

    def set_primitive(self, depth: mi.UInt, value: mi.UInt, active: mi.Bool) -> None:

        self._write_depth_value(self._primitives, depth, value, active)

    def detach_geometry(self) -> None:

        self._vertices_x = dr.detach(self._vertices_x)
        self._vertices_y = dr.detach(self._vertices_y)
        self._vertices_z = dr.detach(self._vertices_z)
        self._theta_t = dr.detach(self._theta_t)
        self._phi_t = dr.detach(self._phi_t)
        self._theta_r = dr.detach(self._theta_r)
        self._phi_r = dr.detach(self._phi_r)

    def _write_depth_value(
        self,
        tensor: mi.TensorXf | mi.TensorXu | mi.TensorXb,
        depth: mi.UInt,
        value: mi.Float | mi.UInt | mi.Bool,
        active: mi.Bool,
    ) -> None:

        valid = active & (depth > 0)
        ind = dr.arange(mi.UInt, self.buffer_size) * self.depth_dim_size + depth - 1
        dr.scatter(tensor.array, value, ind, valid)
