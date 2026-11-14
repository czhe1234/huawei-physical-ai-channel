import mitsuba as mi
import drjit as dr

from .component_buffer import ComponentBuffer
from round2.physical_prior.geometry_prior_engine.constants import (
    InteractionType,
    INVALID_SHAPE,
    INVALID_PRIMITIVE,
)
from round2.physical_prior.geometry_prior_engine.scene import Scene


class PropagationComponents:
    def __init__(
        self,
        scene: Scene,
        component_buffer: ComponentBuffer,
        rel_ant_positions_tx: mi.Point3f,
        rel_ant_positions_rx: mi.Point3f,
    ):

        self._component_buffer = component_buffer

        self._tx_array = scene.tx_array
        self._rx_array = scene.rx_array

        self._num_tx = scene.source_count
        self._num_rx = scene.target_count

        self._wavelength = scene.wavelength

        self._components_built = False

        if component_buffer.buffer_size == 0:
            self._build_empty_components()
            return

        self._eff_tx_array_size = 1
        self._eff_rx_array_size = 1

        src_ind = component_buffer.source_indices
        tgt_ind = component_buffer.target_indices

        self._tx_ind = src_ind // self._eff_tx_array_size
        self._rx_ind = tgt_ind // self._eff_rx_array_size

        self._tx_ant_ind = src_ind % self._eff_tx_array_size
        self._rx_ant_ind = tgt_ind % self._eff_rx_array_size

        num_src = self._num_tx * self._eff_tx_array_size
        num_tgt = self._num_rx * self._eff_rx_array_size
        src_tgt_id = tgt_ind * num_src + src_ind

        num_components = dr.zeros(mi.UInt, num_src * num_tgt)
        self._component_index = dr.scatter_inc(num_components, src_tgt_id)
        dr.eval(self._component_index)
        self._max_components = dr.max(num_components)[0]

        self._build_from_component_buffer()

        self._apply_array_phase_shifts(rel_ant_positions_tx, rel_ant_positions_rx)

        self._fuse_pattern_array_dims()

    @property
    def valid(self):

        return self._valid

    @property
    def a(self):

        return self._a_real, self._a_imag

    @property
    def tau(self):

        return self._tau

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
    def interactions(self):

        if not self._components_built:
            self._assemble_components()
        return self._interactions

    @property
    def objects(self):

        if not self._components_built:
            self._assemble_components()
        return self._shapes

    @property
    def primitives(self):

        if not self._components_built:
            self._assemble_components()
        return self._primitives

    @property
    def vertices(self):

        if not self._components_built:
            self._assemble_components()
        return self._vertices

    def _build_from_component_buffer(self) -> None:

        component_buffer = self._component_buffer
        num_tx = self._num_tx
        num_rx = self._num_rx
        tx_array_size = self._eff_tx_array_size
        rx_array_size = self._eff_rx_array_size
        tx_ind = self._tx_ind
        rx_ind = self._rx_ind
        tx_ant_ind = self._tx_ant_ind
        rx_ant_ind = self._rx_ant_ind
        num_rx_patterns = len(self._rx_array.elements)
        num_tx_patterns = len(self._tx_array.elements)

        component_index = self._component_index

        max_components = self._max_components

        a_real = dr.zeros(
            mi.TensorXf,
            [
                num_rx,
                num_rx_patterns,
                rx_array_size,
                num_tx,
                num_tx_patterns,
                tx_array_size,
                max_components,
            ],
        )
        a_imag = dr.zeros(
            mi.TensorXf,
            [
                num_rx,
                num_rx_patterns,
                rx_array_size,
                num_tx,
                num_tx_patterns,
                tx_array_size,
                max_components,
            ],
        )

        f1 = max_components
        f2 = tx_array_size * f1
        f3 = num_tx_patterns * f2
        f4 = num_tx * f3
        f5 = rx_array_size * f4
        f6 = num_rx_patterns * f5
        response_assignment_index = (
            component_index + tx_ant_ind * f1 + tx_ind * f3 + rx_ant_ind * f4 + rx_ind * f6
        )

        for n in range(num_rx_patterns):
            response_assignment_index_ = dr.copy(response_assignment_index) + n * f5
            for m in range(num_tx_patterns):
                response_assignment_index__ = response_assignment_index_ + m * f2
                dr.scatter(a_real.array, dr.real(component_buffer.a[n][m]), response_assignment_index__)
                dr.scatter(a_imag.array, dr.imag(component_buffer.a[n][m]), response_assignment_index__)

        tensor_shape = [num_rx, num_tx, max_components]
        num_rx_patterns = num_tx_patterns = 1

        f3 = num_tx_patterns * f2
        f4 = num_tx * f3
        f5 = rx_array_size * f4
        f6 = num_rx_patterns * f5
        component_assignment_index = (
            component_index + tx_ant_ind * f1 + tx_ind * f3 + rx_ant_ind * f4 + rx_ind * f6
        )

        valid = dr.full(mi.TensorXb, False, tensor_shape)
        tau = dr.full(mi.TensorXf, -1.0, tensor_shape)
        theta_t = dr.zeros(mi.TensorXf, tensor_shape)
        phi_t = dr.zeros(mi.TensorXf, tensor_shape)
        theta_r = dr.zeros(mi.TensorXf, tensor_shape)
        phi_r = dr.zeros(mi.TensorXf, tensor_shape)

        for n in range(num_rx_patterns):
            component_assignment_index_ = component_assignment_index + n * f5
            for m in range(num_tx_patterns):
                component_assignment_index__ = component_assignment_index_ + m * f2

                dr.scatter(valid.array, True, component_assignment_index__)

                dr.scatter(tau.array, component_buffer.tau, component_assignment_index__)

                dr.scatter(theta_t.array, component_buffer.theta_t, component_assignment_index__)
                dr.scatter(phi_t.array, component_buffer.phi_t, component_assignment_index__)

                dr.scatter(theta_r.array, component_buffer.theta_r, component_assignment_index__)
                dr.scatter(phi_r.array, component_buffer.phi_r, component_assignment_index__)
        self._valid = valid
        self._a_real = a_real
        self._a_imag = a_imag
        self._tau = tau
        self._theta_t = theta_t
        self._phi_t = phi_t
        self._theta_r = theta_r
        self._phi_r = phi_r

    def _apply_array_phase_shifts(
        self, rel_ant_positions_tx: mi.Point3f, rel_ant_positions_rx: mi.Point3f
    ) -> None:

        num_tx = self._num_tx
        num_rx = self._num_rx
        tx_array_size = self._tx_array.array_size
        rx_array_size = self._rx_array.array_size
        max_components = dr.shape(self._a_real)[-1]

        a_real = self._a_real
        a_imag = self._a_imag

        theta_t, phi_t = self._theta_t, self._phi_t

        sin_phi_t, cos_phi_t = dr.sincos(phi_t)
        sin_theta_t, cos_theta_t = dr.sincos(theta_t)
        k_tx_x = sin_theta_t * cos_phi_t
        k_tx_y = sin_theta_t * sin_phi_t
        k_tx_z = cos_theta_t

        k_tx_x = dr.reshape(
            mi.TensorXf, k_tx_x, [num_rx, 1, 1, num_tx, 1, 1, max_components]
        )
        k_tx_y = dr.reshape(
            mi.TensorXf, k_tx_y, [num_rx, 1, 1, num_tx, 1, 1, max_components]
        )
        k_tx_z = dr.reshape(
            mi.TensorXf, k_tx_z, [num_rx, 1, 1, num_tx, 1, 1, max_components]
        )

        theta_r, phi_r = self._theta_r, self._phi_r

        sin_phi_r, cos_phi_r = dr.sincos(phi_r)
        sin_theta_r, cos_theta_r = dr.sincos(theta_r)
        k_rx_x = sin_theta_r * cos_phi_r
        k_rx_y = sin_theta_r * sin_phi_r
        k_rx_z = cos_theta_r

        k_rx_x = dr.reshape(
            mi.TensorXf, k_rx_x, [num_rx, 1, 1, num_tx, 1, 1, max_components]
        )
        k_rx_y = dr.reshape(
            mi.TensorXf, k_rx_y, [num_rx, 1, 1, num_tx, 1, 1, max_components]
        )
        k_rx_z = dr.reshape(
            mi.TensorXf, k_rx_z, [num_rx, 1, 1, num_tx, 1, 1, max_components]
        )

        rel_ant_pos_tx_x = dr.reshape(
            mi.TensorXf, rel_ant_positions_tx.x, [1, 1, 1, num_tx, 1, tx_array_size, 1]
        )
        rel_ant_pos_tx_y = dr.reshape(
            mi.TensorXf, rel_ant_positions_tx.y, [1, 1, 1, num_tx, 1, tx_array_size, 1]
        )
        rel_ant_pos_tx_z = dr.reshape(
            mi.TensorXf, rel_ant_positions_tx.z, [1, 1, 1, num_tx, 1, tx_array_size, 1]
        )

        rel_ant_pos_rx_x = dr.reshape(
            mi.TensorXf, rel_ant_positions_rx.x, [num_rx, 1, rx_array_size, 1, 1, 1, 1]
        )
        rel_ant_pos_rx_y = dr.reshape(
            mi.TensorXf, rel_ant_positions_rx.y, [num_rx, 1, rx_array_size, 1, 1, 1, 1]
        )
        rel_ant_pos_rx_z = dr.reshape(
            mi.TensorXf, rel_ant_positions_rx.z, [num_rx, 1, rx_array_size, 1, 1, 1, 1]
        )

        tx_phase_shifts = rel_ant_pos_tx_z * k_tx_z
        tx_phase_shifts = dr.fma(rel_ant_pos_tx_y, k_tx_y, tx_phase_shifts)
        tx_phase_shifts = dr.fma(rel_ant_pos_tx_x, k_tx_x, tx_phase_shifts)

        rx_phase_shifts = rel_ant_pos_rx_z * k_rx_z
        rx_phase_shifts = dr.fma(rel_ant_pos_rx_y, k_rx_y, rx_phase_shifts)
        rx_phase_shifts = dr.fma(rel_ant_pos_rx_x, k_rx_x, rx_phase_shifts)

        phase_shifts = rx_phase_shifts + tx_phase_shifts
        phase_shifts = dr.two_pi * phase_shifts / self._wavelength

        sin_phi, cos_phi = dr.sincos(phase_shifts)
        a_real_ = dr.fma(a_real, cos_phi, -a_imag * sin_phi)
        a_imag_ = dr.fma(a_real, sin_phi, a_imag * cos_phi)
        self._a_real = a_real_
        self._a_imag = a_imag_

    def _fuse_pattern_array_dims(self) -> None:

        (
            num_rx,
            num_rx_patterns,
            num_rx_ant,
            num_tx,
            num_tx_patterns,
            num_tx_ant,
            max_components,
        ) = dr.shape(self._a_real)

        self._a_real = dr.reshape(
            mi.TensorXf,
            self._a_real,
            [
                num_rx,
                num_rx_patterns * num_rx_ant,
                num_tx,
                num_tx_patterns * num_tx_ant,
                max_components,
            ],
        )
        self._a_imag = dr.reshape(
            mi.TensorXf,
            self._a_imag,
            [
                num_rx,
                num_rx_patterns * num_rx_ant,
                num_tx,
                num_tx_patterns * num_tx_ant,
                max_components,
            ],
        )

    def _build_empty_components(self) -> None:

        max_depth = self._component_buffer.max_depth
        num_tx = self._num_tx
        num_rx = self._num_rx
        num_tx_ant = self._tx_array.num_ant
        num_rx_ant = self._rx_array.num_ant

        a_tensor_base_shape = [num_rx, num_rx_ant, num_tx, num_tx_ant, 0]
        other_tensor_base_shape = [num_rx, num_tx, 0]

        self._valid = dr.full(mi.TensorXb, False, other_tensor_base_shape)
        self._a_real = dr.zeros(mi.TensorXf, a_tensor_base_shape)
        self._a_imag = dr.zeros(mi.TensorXf, a_tensor_base_shape)
        self._tau = dr.zeros(mi.TensorXf, other_tensor_base_shape)
        self._theta_t = dr.zeros(mi.TensorXf, other_tensor_base_shape)
        self._phi_t = dr.zeros(mi.TensorXf, other_tensor_base_shape)
        self._theta_r = dr.zeros(mi.TensorXf, other_tensor_base_shape)
        self._phi_r = dr.zeros(mi.TensorXf, other_tensor_base_shape)

        self._interactions = dr.full(
            mi.TensorXu, InteractionType.NONE, [max_depth] + other_tensor_base_shape
        )
        self._shapes = dr.full(
            mi.TensorXu, INVALID_SHAPE, [max_depth] + other_tensor_base_shape
        )
        self._primitives = dr.full(
            mi.TensorXu, INVALID_PRIMITIVE, [max_depth] + other_tensor_base_shape
        )
        self._vertices = dr.zeros(
            mi.TensorXf, [max_depth] + other_tensor_base_shape + [3]
        )

        self._components_built = True

    def _assemble_components(self) -> None:

        component_buffer = self._component_buffer
        depth_dim_size = component_buffer.depth_dim_size
        total_component_count = component_buffer.buffer_size
        num_tx = self._num_tx
        num_rx = self._num_rx
        tx_array_size = self._eff_tx_array_size
        rx_array_size = self._eff_rx_array_size
        tx_ind = self._tx_ind
        rx_ind = self._rx_ind
        tx_ant_ind = self._tx_ant_ind
        rx_ant_ind = self._rx_ant_ind
        num_rx_patterns = len(self._rx_array.elements)
        num_tx_patterns = len(self._tx_array.elements)

        component_index = self._component_index

        max_components = self._max_components

        tensor_shape = [depth_dim_size, num_rx, num_tx, max_components]
        num_rx_patterns = num_tx_patterns = 1

        depth_ind = dr.arange(mi.UInt, depth_dim_size)

        depth_ind = dr.reshape(mi.TensorXu, depth_ind, [1, depth_dim_size])

        component_index = dr.reshape(mi.TensorXu, component_index, [total_component_count, 1])
        tx_ant_ind = dr.reshape(mi.TensorXu, tx_ant_ind, [total_component_count, 1])
        tx_ind = dr.reshape(mi.TensorXu, tx_ind, [total_component_count, 1])
        rx_ant_ind = dr.reshape(mi.TensorXu, rx_ant_ind, [total_component_count, 1])
        rx_ind = dr.reshape(mi.TensorXu, rx_ind, [total_component_count, 1])

        f1 = max_components
        f2 = tx_array_size * f1
        f3 = num_tx_patterns * f2
        f4 = num_tx * f3
        f5 = rx_array_size * f4
        f6 = num_rx_patterns * f5
        f7 = num_rx * f6

        component_assignment_index = (
            component_index
            + tx_ant_ind * f1
            + tx_ind * f3
            + rx_ant_ind * f4
            + rx_ind * f6
            + depth_ind * f7
        )

        component_assignment_index = component_assignment_index.array

        interactions = dr.full(mi.TensorXu, InteractionType.NONE, tensor_shape)
        shapes = dr.full(mi.TensorXu, INVALID_SHAPE, tensor_shape)
        primitives = dr.full(mi.TensorXu, INVALID_PRIMITIVE, tensor_shape)

        vertices = dr.zeros(mi.TensorXf, tensor_shape + [3])

        for n in range(num_rx_patterns):
            component_assignment_index_ = component_assignment_index + n * f5
            for m in range(num_tx_patterns):
                component_assignment_index__ = component_assignment_index_ + m * f2

                dr.scatter(
                    interactions.array, component_buffer.interaction_types.array, component_assignment_index__
                )

                dr.scatter(shapes.array, component_buffer.shapes.array, component_assignment_index__)

                dr.scatter(primitives.array, component_buffer.primitives.array, component_assignment_index__)

                dr.scatter(
                    vertices.array, component_buffer.vertices_x.array, component_assignment_index__ * 3
                )
                dr.scatter(
                    vertices.array, component_buffer.vertices_y.array, component_assignment_index__ * 3 + 1
                )
                dr.scatter(
                    vertices.array, component_buffer.vertices_z.array, component_assignment_index__ * 3 + 2
                )

        self._interactions = interactions
        self._shapes = shapes
        self._primitives = primitives
        self._vertices = vertices

        self._components_built = True
