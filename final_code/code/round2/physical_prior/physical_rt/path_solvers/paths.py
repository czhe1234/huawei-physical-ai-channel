"""Stores the computed propagation paths"""

import mitsuba as mi
import drjit as dr

from .paths_buffer import PathsBuffer
from round2.physical_prior.physical_rt.constants import InteractionType, INVALID_SHAPE,\
    INVALID_PRIMITIVE
from round2.physical_prior.physical_rt.scene import Scene

class Paths:
    # pylint: disable=line-too-long
    r"""
    Paths()

    Stores the simulated propagation paths

    Paths are generated for the loaded scene using a path solver, such as
    :class:`~round2.physical_prior.physical_rt.PathSolver`. Please refer to the documentation of this
    class for further details.

    :param scene: Scene for which paths are computed
    :param src_positions: Positions of the sources
    :param tgt_positions: Positions of the targets
    :param paths_buffer: Paths buffer storing the computed paths
    :param rel_ant_positions_tx: Positions of the array elements with respect to the center of the transmitters.
    :param rel_ant_positions_rx: Positions of the array elements with respect to the center of the receivers.
    """

    def __init__(self,
                 scene: Scene,
                 src_positions: mi.Point3f,
                 tgt_positions: mi.Point3f,
                 paths_buffer: PathsBuffer,
                 rel_ant_positions_tx: mi.Point3f,
                 rel_ant_positions_rx: mi.Point3f):

        self._paths_buffer = paths_buffer

        # Sources and targets, i.e., end-points of the  paths
        self._src_positions = src_positions
        self._tgt_positions = tgt_positions

        # Referencess to the transmitter and receiver arrays
        self._tx_array = scene.tx_array
        self._rx_array = scene.rx_array

        # Numbers of transmitters and receivers
        self._num_tx = len(scene.transmitters)
        self._num_rx = len(scene.receivers)

        # Wavelength for which the paths were computed
        self._wavelength = scene.wavelength

        # Frequency for which the paths were computed
        self._frequency = scene.frequency

        # Flag indicating if the tensors storing the paths components were built
        # For efficiency, these are only built if requested.
        self._paths_components_built = False

        # To avoid errors, we handle the corner case in which no paths were
        # found separately
        if paths_buffer.buffer_size == 0:
            self._build_empty_paths()
            return # Stop init

        # The following quantities are used in a few internal functions.
        # They are pre-computed here to avoid code duplication

        # One traced endpoint per radio device; array responses are applied
        # afterwards through planar-wave phase shifts.
        self._eff_tx_array_size = 1
        self._eff_rx_array_size = 1

        src_ind = paths_buffer.source_indices
        tgt_ind = paths_buffer.target_indices

        # Indices of transmitters, receivers, and antennas.
        # An antenna-first ordering is assumed.
        self._tx_ind = src_ind // self._eff_tx_array_size
        self._rx_ind = tgt_ind // self._eff_rx_array_size
        #
        self._tx_ant_ind = src_ind % self._eff_tx_array_size
        self._rx_ant_ind = tgt_ind % self._eff_rx_array_size

        # The following uses `dr.scatter_inc` to jointly:
        # - Compute the maximum number of paths over all (source, target)
        # couples, which is then used to allocate the memory for the tensors.
        # - Compute a path index such that paths sharing the same
        # (source, target) couple do not share the same index. This is later
        # used to scatter data in the allocated tensors.
        #
        # We must ensure that the path index used for scattering the path data
        # in the allocated tensors are unique for every (source, target) couple.
        # Map (src_ind, tgt_ind) to a unique integer identifying the path source
        # and target
        num_src = self._num_tx*self._eff_tx_array_size
        num_tgt = self._num_rx*self._eff_rx_array_size
        src_tgt_id = tgt_ind*num_src + src_ind
        #
        num_paths = dr.zeros(mi.UInt, num_src*num_tgt)
        self._path_ind = dr.scatter_inc(num_paths, src_tgt_id)
        dr.eval(self._path_ind)
        self._max_num_paths = dr.max(num_paths)[0]

        # Build the tensors

        # Builds tensors from the paths buffer
        self._build_from_buffer()

        self._apply_array_phase_shifts(rel_ant_positions_tx,
                                       rel_ant_positions_rx)

        # Fuse the pattern and array dimensions
        self._fuse_pattern_array_dims()

    @property
    def sources(self):
        r"""Positions of the transmitter device centers used for tracing.

        :type: :py:class:`mi.Point3f`"""
        return self._src_positions

    @property
    def targets(self):
        r"""Positions of the receiver device centers used for tracing.

        :type: :py:class:`mi.Point3f`"""
        return self._tgt_positions

    @property
    def tx_array(self):
        """Antenna array used by transmitters

        :type: :class:`~round2.physical_prior.physical_rt.AntennaArray`
        """
        return self._tx_array

    @property
    def rx_array(self):
        """Antenna array used by receivers

        :type: :class:`~round2.physical_prior.physical_rt.AntennaArray`"""
        return self._rx_array

    @property
    def num_tx(self):
        """Number of transmitters

        :type: :py:class:`int`"""
        return self._num_tx

    @property
    def num_rx(self):
        """Number of receivers

        :type: :py:class:`int`
        """
        return self._num_rx

    @property
    def valid(self):
        # pylint: disable=line-too-long
        """
        Flags indicating valid paths

        :type: :py:class:`mi.TensorXb [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]`
        """
        return self._valid

    @property
    def a(self):
        # pylint: disable=line-too-long
        r"""
        Real and imaginary components of the channel coefficients [unitless, linear scale]

        :type: :py:class:`Tuple[mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]]`
        """
        return self._a_real, self._a_imag

    @property
    def tau(self):
        # pylint: disable=line-too-long
        """
        Paths delays [s]

        :type: :py:class:`mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]`
        """
        return self._tau

    @property
    def theta_t(self):
        # pylint: disable=line-too-long
        """
        Zenith  angles of departure [rad]

        :type: :py:class:`mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]`
        """
        return self._theta_t

    @property
    def phi_t(self):
        # pylint: disable=line-too-long
        """
        Azimuth  angles of departure [rad]

        :type: :py:class:`mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]`
        """
        return self._phi_t

    @property
    def theta_r(self):
        # pylint: disable=line-too-long
        """
        Zenith  angles of arrival [rad]

        :type: :py:class:`mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]`
        """
        return self._theta_r

    @property
    def phi_r(self):
        # pylint: disable=line-too-long
        """
        Azimuth  angles of arrival [rad]

        :type: :py:class:`mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]`
        """
        return self._phi_r

    @property
    def interactions(self):
        # pylint: disable=line-too-long
        """
        Interaction type represented using
        :class:`~round2.physical_prior.physical_rt.constants.InteractionType`

        :type: :py:class:`mi.TensorXu [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [max_depth, num_rx, num_tx, num_paths]`
        """
        if not self._paths_components_built:
            self._build_paths_components()
        return self._interactions

    @property
    def objects(self):
        # pylint: disable=line-too-long
        """
        IDs of the intersected objects. Invalid objects are represented by
        :data:`~round2.physical_prior.physical_rt.constants.INVALID_SHAPE`.

        :type: :py:class:`mi.TensorXu [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [max_depth, num_rx, num_tx, num_paths]`
        """
        if not self._paths_components_built:
            self._build_paths_components()
        return self._shapes

    @property
    def primitives(self):
        # pylint: disable=line-too-long
        """
        Indices of the intersected primitives. Invalid primitives are
        represented by :data:`~round2.physical_prior.physical_rt.constants.INVALID_PRIMITIVE`.

        :type: :py:class:`mi.TensorXu [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [max_depth, num_rx, num_tx, num_paths]`
        """
        if not self._paths_components_built:
            self._build_paths_components()
        return self._primitives

    @property
    def vertices(self):
        # pylint: disable=line-too-long
        """
        Paths' vertices, i.e., the interaction points of the paths with the
        scene

        :type: :py:class:`mi.TensorXf [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, 3] or [max_depth, num_rx, num_tx, num_paths, 3]`
        """
        if not self._paths_components_built:
            self._build_paths_components()
        return self._vertices

    def _build_from_buffer(self) -> None:
        r"""
        Builds the multi-dimensional tensors storing the paths data from the
        path buffer object

        This function reorganizes the path data (coefficients, delays, angles
        of arrival and departure) into tensors.

        The vertices, intersected objects and primitives and other data only
        required for rendering or for more exhaustive study of the paths are not
        processed by this function and only computed on demand.
        """

        paths_buffer = self._paths_buffer
        num_tx = self._num_tx
        num_rx = self._num_rx
        tx_array_size = self._eff_tx_array_size
        rx_array_size = self._eff_rx_array_size
        tx_ind = self._tx_ind
        rx_ind = self._rx_ind
        tx_ant_ind = self._tx_ant_ind
        rx_ant_ind = self._rx_ant_ind
        num_rx_patterns = len(self._rx_array.antenna_pattern.patterns)
        num_tx_patterns = len(self._tx_array.antenna_pattern.patterns)
        # Path index such that paths sharing the same (source, target) couple
        # do not share the same index.
        path_ind = self._path_ind
        # Maximum number of paths over all (source, target) couples
        max_num_paths = self._max_num_paths

        # Allocate the tensors and fill them
        # `a`
        a_real = dr.zeros(mi.TensorXf, [num_rx, num_rx_patterns, rx_array_size,
                                        num_tx, num_tx_patterns, tx_array_size,
                                        max_num_paths])
        a_imag = dr.zeros(mi.TensorXf, [num_rx, num_rx_patterns, rx_array_size,
                                        num_tx, num_tx_patterns, tx_array_size,
                                        max_num_paths])
        # Scatter indices
        f1 = max_num_paths
        f2 = tx_array_size*f1
        f3 = num_tx_patterns*f2
        f4 = num_tx*f3
        f5 = rx_array_size*f4
        f6 = num_rx_patterns*f5
        a_scat_ind = path_ind + tx_ant_ind*f1 + tx_ind*f3 + rx_ant_ind*f4\
                        + rx_ind*f6
        # Fill the `a` tensor
        for n in range(num_rx_patterns):
            a_scat_ind_ = dr.copy(a_scat_ind) + n*f5
            for m in range(num_tx_patterns):
                a_scat_ind__ = a_scat_ind_ + m*f2
                dr.scatter(a_real.array, dr.real(paths_buffer.a[n][m]),
                        a_scat_ind__)
                dr.scatter(a_imag.array, dr.imag(paths_buffer.a[n][m]),
                        a_scat_ind__)


        # Geometry and angle tensors are traced per device pair.
        tensor_shape = [num_rx, num_tx, max_num_paths]
        num_rx_patterns = num_tx_patterns = 1

        # We need to recompute these factors as `num_rx_patterns`
        # or `num_rx_patterns` might have changed
        f3 = num_tx_patterns*f2
        f4 = num_tx*f3
        f5 = rx_array_size*f4
        f6 = num_rx_patterns*f5
        scat_ind = path_ind + tx_ant_ind*f1 + tx_ind*f3 + rx_ant_ind*f4\
                    + rx_ind*f6

        # Instantiate the tensors
        valid = dr.full(mi.TensorXb, False, tensor_shape)
        tau = dr.full(mi.TensorXf, -1., tensor_shape)
        theta_t = dr.zeros(mi.TensorXf, tensor_shape)
        phi_t = dr.zeros(mi.TensorXf, tensor_shape)
        theta_r = dr.zeros(mi.TensorXf, tensor_shape)
        phi_r = dr.zeros(mi.TensorXf, tensor_shape)
        # Fill the tensor
        # Note that we cannot fuse this loop with the one used to fill `a` as
        # `num_rx_patterns` or `num_tx_patterns` might be different
        for n in range(num_rx_patterns):
            scat_ind_ = scat_ind + n*f5
            for m in range(num_tx_patterns):
                scat_ind__ = scat_ind_ + m*f2
                #
                dr.scatter(valid.array, True, scat_ind__)
                #
                dr.scatter(tau.array, paths_buffer.tau, scat_ind__)
                #
                dr.scatter(theta_t.array, paths_buffer.theta_t, scat_ind__)
                dr.scatter(phi_t.array, paths_buffer.phi_t, scat_ind__)
                #
                dr.scatter(theta_r.array, paths_buffer.theta_r, scat_ind__)
                dr.scatter(phi_r.array, paths_buffer.phi_r, scat_ind__)
        self._valid = valid
        self._a_real = a_real
        self._a_imag = a_imag
        self._tau = tau
        self._theta_t = theta_t
        self._phi_t = phi_t
        self._theta_r = theta_r
        self._phi_r = phi_r

    def _apply_array_phase_shifts(self,
                                  rel_ant_positions_tx: mi.Point3f,
                                  rel_ant_positions_rx: mi.Point3f) -> None:
        r"""
        Applies planar-wave phase shifts for all array elements.

        :param rel_ant_positions_tx: Positions of the array elements with
            respect to the center of the transmitters.

        :param rel_ant_positions_rx: Positions of the array elements with
            respect to the center of the receivers.
        """

        num_tx = self._num_tx
        num_rx = self._num_rx
        tx_array_size = self._tx_array.array_size
        rx_array_size = self._rx_array.array_size
        max_num_paths = dr.shape(self._a_real)[-1]

        # [num_rx, num_rx_patterns, 1, num_tx, num_tx_patterns, 1,
        #   max_num_paths]
        a_real = self._a_real
        a_imag = self._a_imag

        # Directions of arrival and departures
        # [num_rx, num_tx, max_num_paths]
        theta_t, phi_t = self._theta_t, self._phi_t
        # [num_tx, num_tx, max_num_paths, 3]
        sin_phi_t, cos_phi_t = dr.sincos(phi_t)
        sin_theta_t, cos_theta_t = dr.sincos(theta_t)
        k_tx_x = sin_theta_t*cos_phi_t
        k_tx_y = sin_theta_t*sin_phi_t
        k_tx_z = cos_theta_t
        # Expand for broadcasting
        k_tx_x = dr.reshape(mi.TensorXf, k_tx_x,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])
        k_tx_y = dr.reshape(mi.TensorXf, k_tx_y,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])
        k_tx_z = dr.reshape(mi.TensorXf, k_tx_z,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])
        # [num_rx, num_tx, max_num_paths]
        theta_r, phi_r = self._theta_r, self._phi_r
        # [num_tx, num_tx, max_num_paths, 3]
        sin_phi_r, cos_phi_r = dr.sincos(phi_r)
        sin_theta_r, cos_theta_r = dr.sincos(theta_r)
        k_rx_x = sin_theta_r*cos_phi_r
        k_rx_y = sin_theta_r*sin_phi_r
        k_rx_z = cos_theta_r
        # Expand for broadcasting
        k_rx_x = dr.reshape(mi.TensorXf, k_rx_x,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])
        k_rx_y = dr.reshape(mi.TensorXf, k_rx_y,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])
        k_rx_z = dr.reshape(mi.TensorXf, k_rx_z,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])

        # Relative positions of the antennas of the transmitters and receivers
        # [1, 1, 1, num_tx, 1, tx_array_size, 1]
        rel_ant_pos_tx_x = dr.reshape(mi.TensorXf, rel_ant_positions_tx.x,
                                      [1, 1, 1, num_tx, 1, tx_array_size, 1])
        rel_ant_pos_tx_y = dr.reshape(mi.TensorXf, rel_ant_positions_tx.y,
                                      [1, 1, 1, num_tx, 1, tx_array_size, 1])
        rel_ant_pos_tx_z = dr.reshape(mi.TensorXf, rel_ant_positions_tx.z,
                                      [1, 1, 1, num_tx, 1, tx_array_size, 1])

        # Relative positions of the antennas of the transmitters and receivers
        # [num_rx, 1, rx_array_size, 1, 1, 1, 1]
        rel_ant_pos_rx_x = dr.reshape(mi.TensorXf, rel_ant_positions_rx.x,
                                      [num_rx, 1, rx_array_size, 1, 1, 1, 1])
        rel_ant_pos_rx_y = dr.reshape(mi.TensorXf, rel_ant_positions_rx.y,
                                      [num_rx, 1, rx_array_size, 1, 1, 1, 1])
        rel_ant_pos_rx_z = dr.reshape(mi.TensorXf, rel_ant_positions_rx.z,
                                      [num_rx, 1, rx_array_size, 1, 1, 1, 1])

        # Compute the phase shifts by taking the dot products between directions
        # off departure (arrival) and the antenna array relative positions
        # TX
        # [num_rx, 1, 1, num_tx, 1, tx_array_size, max_num_paths]
        tx_phase_shifts = rel_ant_pos_tx_z*k_tx_z
        tx_phase_shifts = dr.fma(rel_ant_pos_tx_y, k_tx_y, tx_phase_shifts)
        tx_phase_shifts = dr.fma(rel_ant_pos_tx_x, k_tx_x, tx_phase_shifts)
        # RX
        # [num_rx, 1, rx_array_size, num_tx, 1, 1, max_num_paths]
        rx_phase_shifts = rel_ant_pos_rx_z*k_rx_z
        rx_phase_shifts = dr.fma(rel_ant_pos_rx_y, k_rx_y, rx_phase_shifts)
        rx_phase_shifts = dr.fma(rel_ant_pos_rx_x, k_rx_x, rx_phase_shifts)
        # Total phase shift
        # [num_rx, 1, rx_array_size, num_tx, 1, tx_array_size, max_num_paths]
        phase_shifts = rx_phase_shifts + tx_phase_shifts
        phase_shifts = dr.two_pi*phase_shifts/self._wavelength

        # Apply the phase shifts
        # [num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns,
        # tx_array_size, max_num_paths]
        sin_phi, cos_phi = dr.sincos(phase_shifts)
        a_real_ = dr.fma(a_real, cos_phi, -a_imag*sin_phi)
        a_imag_ = dr.fma(a_real, sin_phi, a_imag*cos_phi)
        self._a_real = a_real_
        self._a_imag = a_imag_

    def _fuse_pattern_array_dims(self) -> None:
        r"""
        Merges the pattern and array dimentions of the tensors storing the
        channel coefficients
        """

        num_rx, num_rx_patterns, num_rx_ant, num_tx, num_tx_patterns,\
            num_tx_ant, max_num_paths = dr.shape(self._a_real)

        # [num_rx, num_rx_patterns*rx_array_size, num_tx,
        # num_tx_patterns*tx_array_size, max_num_paths]
        self._a_real = dr.reshape(mi.TensorXf, self._a_real,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])
        self._a_imag = dr.reshape(mi.TensorXf, self._a_imag,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])

    def _build_empty_paths(self) -> None:
        r"""
        Builds empty tensors with dimensions fitting the setting of the paths
        buffer, i.e., same number of radio devices, antennas, etc
        """

        max_depth = self._paths_buffer.max_depth
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
        #
        self._interactions = dr.full(mi.TensorXu, InteractionType.NONE,
                                     [max_depth] + other_tensor_base_shape)
        self._shapes = dr.full(mi.TensorXu, INVALID_SHAPE,
                               [max_depth] + other_tensor_base_shape)
        self._primitives = dr.full(mi.TensorXu, INVALID_PRIMITIVE,
                                   [max_depth] + other_tensor_base_shape)
        self._vertices = dr.zeros(mi.TensorXf,
                                  [max_depth] + other_tensor_base_shape + [3])

        self._paths_components_built = True

    def _build_paths_components(self) -> None:
        r"""
        Builds and fills tensors for storing the additional paths data required
        for paths visualization or more exhaustive studies of the paths
        (vertices, interaction types, etc)
        """

        paths_buffer = self._paths_buffer
        depth_dim_size = paths_buffer.depth_dim_size
        total_paths_count = paths_buffer.buffer_size
        num_tx = self._num_tx
        num_rx = self._num_rx
        tx_array_size = self._eff_tx_array_size
        rx_array_size = self._eff_rx_array_size
        tx_ind = self._tx_ind
        rx_ind = self._rx_ind
        tx_ant_ind = self._tx_ant_ind
        rx_ant_ind = self._rx_ant_ind
        num_rx_patterns = len(self._rx_array.antenna_pattern.patterns)
        num_tx_patterns = len(self._tx_array.antenna_pattern.patterns)
        # Path index such that paths sharing the same (source, target) couple
        # do not share the same index.
        path_ind = self._path_ind
        # Maximum number of paths over all (source, target) couples
        max_num_paths = self._max_num_paths

        tensor_shape = [depth_dim_size, num_rx, num_tx, max_num_paths]
        num_rx_patterns = num_tx_patterns = 1

        # To build the indices that are used for scattering in the tensors, we
        # reshape the path, device, and antenna indices to tensors to leverage
        # the broadcasting feature of dr.jit
        # [depth_dim_size]
        depth_ind = dr.arange(mi.UInt, depth_dim_size)
        # [1, depth_dim_size]
        depth_ind = dr.reshape(mi.TensorXu, depth_ind, [1, depth_dim_size])
        # [total_paths_count, 1]
        path_ind = dr.reshape(mi.TensorXu, path_ind, [total_paths_count, 1])
        tx_ant_ind = dr.reshape(mi.TensorXu, tx_ant_ind, [total_paths_count, 1])
        tx_ind = dr.reshape(mi.TensorXu, tx_ind, [total_paths_count, 1])
        rx_ant_ind = dr.reshape(mi.TensorXu, rx_ant_ind, [total_paths_count, 1])
        rx_ind = dr.reshape(mi.TensorXu, rx_ind, [total_paths_count, 1])
        # Scatter indices
        f1 = max_num_paths
        f2 = tx_array_size*f1
        f3 = num_tx_patterns*f2
        f4 = num_tx*f3
        f5 = rx_array_size*f4
        f6 = num_rx_patterns*f5
        f7 = num_rx*f6
        # [total_paths_count, depth_dim_size]
        scat_ind = path_ind + tx_ant_ind*f1 + tx_ind*f3 + rx_ant_ind*f4\
                    + rx_ind*f6 + depth_ind*f7
        # [total_paths_count*depth_dim_size]
        scat_ind = scat_ind.array

        # Allocate and fill the tensors

        interactions = dr.full(mi.TensorXu, InteractionType.NONE, tensor_shape)
        shapes = dr.full(mi.TensorXu, INVALID_SHAPE, tensor_shape)
        primitives = dr.full(mi.TensorXu, INVALID_PRIMITIVE, tensor_shape)
        # `vertices` requires an extra dimension for storing the (x,y,z)
        # coordinates
        vertices = dr.zeros(mi.TensorXf, tensor_shape + [3])

        for n in range(num_rx_patterns):
            scat_ind_ = scat_ind + n*f5
            for m in range(num_tx_patterns):
                scat_ind__ = scat_ind_ + m*f2
                #
                dr.scatter(interactions.array,
                           paths_buffer.interaction_types.array,
                           scat_ind__)
                #
                dr.scatter(shapes.array, paths_buffer.shapes.array, scat_ind__)
                #
                dr.scatter(primitives.array, paths_buffer.primitives.array,
                           scat_ind__)
                #
                dr.scatter(vertices.array, paths_buffer.vertices_x.array,
                           scat_ind__*3)
                dr.scatter(vertices.array, paths_buffer.vertices_y.array,
                           scat_ind__*3 + 1)
                dr.scatter(vertices.array, paths_buffer.vertices_z.array,
                           scat_ind__*3 + 2)

        self._interactions = interactions
        self._shapes = shapes
        self._primitives = primitives
        self._vertices = vertices

        self._paths_components_built = True
